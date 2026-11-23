// RUN: not llvm-tblgen -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s
// RUN: not llvm-tblgen -DERROR2 %s 2>&1 | FileCheck --check-prefix=ERROR2 %s

#ifdef ERROR1
// ERROR1: [[@LINE+1]]:9: error: Unknown token when expecting a type
class C<field int x = 0>;
#endif

#ifdef ERROR2
// ERROR2: [[@LINE+1]]:14: error: Unknown token when expecting a type
multiclass M<field string s> {
  def D;
}
#endif
