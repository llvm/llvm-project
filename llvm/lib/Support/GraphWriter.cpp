//===-- GraphWriter.cpp - Implements GraphWriter support routines ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements misc. GraphWriter support routines.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Config/config.h"
using namespace llvm;

static cl::opt<bool> ViewBackground("view-background", cl::Hidden,
  cl::desc("Execute graph viewer in the background. Creates tmp file litter."));

std::string llvm::DOT::EscapeString(const std::string &Label) {
  std::string Str(Label);
  for (unsigned i = 0; i != Str.length(); ++i)
  switch (Str[i]) {
    case '\n':
      Str.insert(Str.begin()+i, '\\');  // Escape character...
      ++i;
      Str[i] = 'n';
      break;
    case '\t':
      Str.insert(Str.begin()+i, ' ');  // Convert to two spaces
      ++i;
      Str[i] = ' ';
      break;
    case '\\':
      if (i+1 != Str.length())
        switch (Str[i+1]) {
          case 'l': continue; // don't disturb \l
          case '|': case '{': case '}':
            Str.erase(Str.begin()+i); continue;
          default: break;
        }
    case '{': case '}':
    case '<': case '>':
    case '|': case '"':
      Str.insert(Str.begin()+i, '\\');  // Escape character...
      ++i;  // don't infinite loop
      break;
  }
  return Str;
}

// Execute the graph viewer. Return true if successful.
static bool LLVM_ATTRIBUTE_UNUSED
ExecGraphViewer(const sys::Path &ExecPath, std::vector<const char*> &args,
                const sys::Path &Filename, bool wait, std::string &ErrMsg) {
  if (wait) {
    if (sys::Program::ExecuteAndWait(ExecPath, &args[0],0,0,0,0,&ErrMsg)) {
      errs() << "Error: " << ErrMsg << "\n";
      return false;
    }
    Filename.eraseFromDisk();
    errs() << " done. \n";
  }
  else {
    sys::Program::ExecuteNoWait(ExecPath, &args[0],0,0,0,&ErrMsg);
    errs() << "Remember to erase graph file: " << Filename.str() << "\n";
  }
  return true;
}

void llvm::DisplayGraph(const sys::Path &Filename, bool wait,
                        GraphProgram::Name program) {
  wait &= !ViewBackground;
  std::string ErrMsg;
#if HAVE_GRAPHVIZ
  sys::Path Graphviz(LLVM_PATH_GRAPHVIZ);

  std::vector<const char*> args;
  args.push_back(Graphviz.c_str());
  args.push_back(Filename.c_str());
  args.push_back(0);

  errs() << "Running 'Graphviz' program... ";
  if (!ExecGraphViewer(Graphviz, args, Filename, wait, ErrMsg))
    return;

#elif HAVE_XDOT_PY
  std::vector<const char*> args;
  args.push_back(LLVM_PATH_XDOT_PY);
  args.push_back(Filename.c_str());

  switch (program) {
  case GraphProgram::DOT:   args.push_back("-f"); args.push_back("dot"); break;
  case GraphProgram::FDP:   args.push_back("-f"); args.push_back("fdp"); break;
  case GraphProgram::NEATO: args.push_back("-f"); args.push_back("neato");break;
  case GraphProgram::TWOPI: args.push_back("-f"); args.push_back("twopi");break;
  case GraphProgram::CIRCO: args.push_back("-f"); args.push_back("circo");break;
  }

  args.push_back(0);

  errs() << "Running 'xdot.py' program... ";
  if (!ExecGraphViewer(sys::Path(LLVM_PATH_XDOT_PY), args, Filename, wait, ErrMsg))
    return;

#elif (HAVE_GV && (HAVE_DOT || HAVE_FDP || HAVE_NEATO || \
                   HAVE_TWOPI || HAVE_CIRCO))
  sys::Path PSFilename = Filename;
  PSFilename.appendSuffix("ps");

  sys::Path prog;

  // Set default grapher
#if HAVE_CIRCO
  prog = sys::Path(LLVM_PATH_CIRCO);
#endif
#if HAVE_TWOPI
  prog = sys::Path(LLVM_PATH_TWOPI);
#endif
#if HAVE_NEATO
  prog = sys::Path(LLVM_PATH_NEATO);
#endif
#if HAVE_FDP
  prog = sys::Path(LLVM_PATH_FDP);
#endif
#if HAVE_DOT
  prog = sys::Path(LLVM_PATH_DOT);
#endif

  // Find which program the user wants
#if HAVE_DOT
  if (program == GraphProgram::DOT)
    prog = sys::Path(LLVM_PATH_DOT);
#endif
#if (HAVE_FDP)
  if (program == GraphProgram::FDP)
    prog = sys::Path(LLVM_PATH_FDP);
#endif
#if (HAVE_NEATO)
  if (program == GraphProgram::NEATO)
    prog = sys::Path(LLVM_PATH_NEATO);
#endif
#if (HAVE_TWOPI)
  if (program == GraphProgram::TWOPI)
    prog = sys::Path(LLVM_PATH_TWOPI);
#endif
#if (HAVE_CIRCO)
  if (program == GraphProgram::CIRCO)
    prog = sys::Path(LLVM_PATH_CIRCO);
#endif

  std::vector<const char*> args;
  args.push_back(prog.c_str());
  args.push_back("-Tps");
  args.push_back("-Nfontname=Courier");
  args.push_back("-Gsize=7.5,10");
  args.push_back(Filename.c_str());
  args.push_back("-o");
  args.push_back(PSFilename.c_str());
  args.push_back(0);

  errs() << "Running '" << prog.str() << "' program... ";

  if (!ExecGraphViewer(prog, args, Filename, wait, ErrMsg))
    return;

  sys::Path gv(LLVM_PATH_GV);
  args.clear();
  args.push_back(gv.c_str());
  args.push_back(PSFilename.c_str());
  args.push_back("--spartan");
  args.push_back(0);

  ErrMsg.clear();
  if (!ExecGraphViewer(gv, args, PSFilename, wait, ErrMsg))
    return;

#elif HAVE_DOTTY
  sys::Path dotty(LLVM_PATH_DOTTY);

  std::vector<const char*> args;
  args.push_back(dotty.c_str());
  args.push_back(Filename.c_str());
  args.push_back(0);

// Dotty spawns another app and doesn't wait until it returns
#if defined (__MINGW32__) || defined (_WINDOWS)
  wait = false;
#endif
  errs() << "Running 'dotty' program... ";
  if (!ExecGraphViewer(dotty, args, Filename, wait, ErrMsg))
    return;
#endif
}
