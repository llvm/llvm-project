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

#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/Streams.h"
#include "llvm/System/Path.h"
#include "llvm/System/Program.h"
#include "llvm/Config/config.h"
using namespace llvm;

void llvm::DisplayGraph(const sys::Path &Filename) {
  std::string ErrMsg;
#if HAVE_GRAPHVIZ
  sys::Path Graphviz(LLVM_PATH_GRAPHVIZ);

  std::vector<const char*> args;
  args.push_back(Graphviz.c_str());
  args.push_back(Filename.c_str());
  args.push_back(0);
  
  cerr << "Running 'Graphviz' program... " << std::flush;
  if (sys::Program::ExecuteAndWait(Graphviz, &args[0],0,0,0,0,&ErrMsg)) {
    cerr << "Error viewing graph: " << ErrMsg << "\n";
  }
#elif (HAVE_GV && HAVE_DOT)
  sys::Path PSFilename = Filename;
  PSFilename.appendSuffix("ps");
  
  sys::Path dot(LLVM_PATH_DOT);

  std::vector<const char*> args;
  args.push_back(dot.c_str());
  args.push_back("-Tps");
  args.push_back("-Nfontname=Courier");
  args.push_back("-Gsize=7.5,10");
  args.push_back(Filename.c_str());
  args.push_back("-o");
  args.push_back(PSFilename.c_str());
  args.push_back(0);
  
  cerr << "Running 'dot' program... " << std::flush;
  if (sys::Program::ExecuteAndWait(dot, &args[0],0,0,0,0,&ErrMsg)) {
    cerr << "Error viewing graph: '" << ErrMsg << "\n";
  } else {
    cerr << " done. \n";

    sys::Path gv(LLVM_PATH_GV);
    args.clear();
    args.push_back(gv.c_str());
    args.push_back(PSFilename.c_str());
    args.push_back("-spartan");
    args.push_back(0);
    
    ErrMsg.clear();
    if (sys::Program::ExecuteAndWait(gv, &args[0],0,0,0,0,&ErrMsg)) {
      cerr << "Error viewing graph: " << ErrMsg << "\n";
    }
  }
  PSFilename.eraseFromDisk();
#elif HAVE_DOTTY
  sys::Path dotty(LLVM_PATH_DOTTY);

  std::vector<const char*> args;
  args.push_back(dotty.c_str());
  args.push_back(Filename.c_str());
  args.push_back(0);
  
  cerr << "Running 'dotty' program... " << std::flush;
  if (sys::Program::ExecuteAndWait(dotty, &args[0],0,0,0,0,&ErrMsg)) {
    cerr << "Error viewing graph: " << ErrMsg << "\n";
  } else {
#ifdef __MINGW32__ // Dotty spawns another app and doesn't wait until it returns
    return;
#endif
  }
#endif
  
  Filename.eraseFromDisk();
}
