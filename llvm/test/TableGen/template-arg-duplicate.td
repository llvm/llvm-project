// RUN: not llvm-tblgen -DCLASS %s 2>&1 | FileCheck %s --check-prefix=CLASS
// RUN: not llvm-tblgen -DMULTICLASS %s 2>&1 | FileCheck %s --check-prefix=MULTICLASS
// RUN: not llvm-tblgen -DTYPE %s 2>&1 | FileCheck %s --check-prefix=TYPE --implicit-check-not=error:

// Check that a duplicate template argument name is diagnosed at the start of
// the second declaration.

#ifdef CLASS
// CLASS: [[@LINE+1]]:16: error: template argument with the same name has already been defined
class C<int a, int a>;
#endif

#ifdef MULTICLASS
// MULTICLASS: [[@LINE+1]]:21: error: template argument with the same name has already been defined
multiclass M<int a, int a> {
  def NAME;
}
#endif

// A duplicate with a different type and a default value only gives one error.
#ifdef TYPE
// TYPE: [[@LINE+1]]:16: error: template argument with the same name has already been defined
class D<int a, string a = "x">;
#endif
