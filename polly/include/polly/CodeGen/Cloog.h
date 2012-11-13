//===- CLooG.h - CLooG interface --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// CLooG[1] interface.
//
// The CLooG interface takes a Scop and generates a CLooG AST (clast). This
// clast can either be returned directly or it can be pretty printed to stdout.
//
// A typical clast output looks like this:
//
// for (c2 = max(0, ceild(n + m, 2); c2 <= min(511, floord(5 * n, 3)); c2++) {
//   bb2(c2);
// }
//
// [1] http://www.cloog.org/ - The Chunky Loop Generator
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_CLOOG_H
#define POLLY_CLOOG_H

#include "polly/Config/config.h"
#ifdef CLOOG_FOUND

#include "polly/ScopPass.h"

#define CLOOG_INT_GMP 1
#include "cloog/cloog.h"

struct clast_name;
namespace llvm {
  class raw_ostream;
}

namespace polly {
  class Scop;
  class Cloog;

  class CloogInfo : public ScopPass {
    Cloog *C;
    Scop *scop;

  public:
    static char ID;
    CloogInfo() : ScopPass(ID), C(0) {}

    /// Write a .cloog input file
    void dump(FILE *F);

    /// Print a source code representation of the program.
    void pprint(llvm::raw_ostream &OS);

    /// Create the CLooG AST from this program.
    const struct clast_root *getClast();

    bool runOnScop(Scop &S);
    void printScop(llvm::raw_ostream &OS) const;
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual void releaseMemory();
  };

  // Visitor class for clasts.
  // Only 'visitUser' has to be implemented by subclasses; the default
  // implementations of the other methods traverse the clast recursively.
  class ClastVisitor {
  public:
    virtual void visit(const clast_stmt *stmt);

    virtual void visitAssignment(const clast_assignment *stmt);
    virtual void visitBlock(const clast_block *stmt);
    virtual void visitFor(const clast_for *stmt);
    virtual void visitGuard(const clast_guard *stmt);

    virtual void visitUser(const clast_user_stmt *stmt) = 0;
  };
}

namespace llvm {
  class PassRegistry;
  void initializeCloogInfoPass(llvm::PassRegistry&);
}

#endif /* CLOOG_FOUND */
#endif /* POLLY_CLOOG_H */
