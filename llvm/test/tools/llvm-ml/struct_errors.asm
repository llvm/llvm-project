; RUN: not llvm-ml -filetype=s %s /Fo /dev/null 2>&1 | FileCheck %s --dump-input=always

.data
int_test STRUCT
  int_arr DWORD ?, ?
  int_scalar DWORD ?
int_test ENDS

t1 int_test <<1,2,3>>
; CHECK: error: Initializer too long for field; expected at most 2 elements, got 3

t2 int_test <4>
; CHECK: error: Cannot initialize array field with scalar value

t3 int_test <,<5,6>>
; CHECK: error: Cannot initialize scalar field with array value

real_test STRUCT
  real_arr REAL4 ?, ?, ?
  real_scalar REAL4 ?
real_test ENDS

t4 real_test <<1.0,0.0,-1.0,-2.0>>
; CHECK: error: Initializer too long for field; expected at most 3 elements, got 4

t5 real_test <2.0>
; CHECK: error: Cannot initialize array field with scalar value

t6 real_test <,<2.0,-2.0>>
; CHECK: error: Cannot initialize scalar field with array value

inner_struct STRUCT
  a BYTE ?
inner_struct ENDS

struct_test STRUCT
  struct_arr inner_struct 4 DUP (?)
  struct_scalar inner_struct ?
struct_test ENDS

t7 struct_test <<<>, <>, <>, <>, <>>>
; CHECK: error: Initializer too long for field; expected at most 4 elements, got 5

t8 struct_test <,<<>, <>>>
; CHECK: error: 'inner_struct' initializer initializes too many fields

t9 STRUCT 3
; CHECK: error: alignment must be a power of two; was 3
t9 ENDS

t10 STRUCT 1, X
; CHECK: error: Unrecognized qualifier for 'STRUCT' directive; expected none or NONUNIQUE
t10 ENDS

t11 STRUCT
different_struct ENDS
; CHECK: error: mismatched name in ENDS directive; expected 't11'
