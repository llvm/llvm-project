#include "pseudo_barrier.h"

#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBModule.h"
#include "lldb/API/SBSymbol.h"
#include "lldb/API/SBTarget.h"

#include "common.h"

using namespace lldb;

// The demangled name of a symbol (and the info describing which parts of it are
// the basename, the arguments, etc.) is computed on first access and cached in
// the Mangled object of that symbol. SBSymbol::GetName() takes no lock, so
// several threads can run that lazy computation for the same symbol at once.
// This used to hand out and free the cached info from several threads at the
// same time, which crashed with a double free.
void test(SBDebugger &dbg, std::vector<std::string> args) {
  dbg.SetAsync(false);
  SBTarget target = dbg.CreateTarget(args.at(0).c_str());
  if (!target.IsValid())
    throw Exception("Invalid target");

  // Keep the barrier waits below (which spin) bounded on targets with a lot of
  // symbols.
  const size_t max_symbols = 1000;

  // Collect the symbols without asking for their names, so that the threads
  // below are the first ones to demangle them.
  std::vector<SBSymbol> symbols;
  for (uint32_t i = 0; i < target.GetNumModules() && symbols.size() < max_symbols;
       ++i) {
    SBModule module = target.GetModuleAtIndex(i);
    for (uint32_t j = 0;
         j < module.GetNumSymbols() && symbols.size() < max_symbols; ++j)
      symbols.push_back(module.GetSymbolAtIndex(j));
  }
  if (symbols.empty())
    throw Exception("Target has no symbols");

  const size_t num_threads = 5;

  // Rendezvous the threads before each individual symbol so that they all
  // demangle the same symbol at the same time.
  std::unique_ptr<pseudo_barrier_t[]> barriers(
      new pseudo_barrier_t[symbols.size()]);
  for (size_t i = 0; i < symbols.size(); ++i)
    pseudo_barrier_init(barriers[i], num_threads);

  // The names each thread saw, so that we can check that they all agree.
  std::vector<std::vector<const char *>> names(num_threads);

  auto lambda = [&](size_t thread_index) {
    std::vector<const char *> &thread_names = names[thread_index];
    thread_names.reserve(symbols.size());
    for (size_t i = 0; i < symbols.size(); ++i) {
      pseudo_barrier_wait(barriers[i]);
      thread_names.push_back(symbols[i].GetName());
    }
  };

  std::vector<std::thread> threads;
  for (size_t i = 0; i < num_threads; ++i)
    threads.emplace_back(lambda, i);
  for (std::thread &thread : threads)
    thread.join();

  // Whichever thread won the race, all of them should have seen the same name.
  for (size_t i = 0; i < symbols.size(); ++i) {
    const char *expected = names[0][i];
    for (size_t t = 1; t < num_threads; ++t) {
      const char *name = names[t][i];
      if (!expected != !name || (expected && ::strcmp(expected, name) != 0))
        throw Exception("Threads disagree about the name of a symbol");
    }
  }
}
