; RUN: not llvm-ml -filetype=s %s /Fo /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

.data
org_struct STRUCT
  x BYTE ?
  x_succ BYTE ?
  ORG 15
  y BYTE ?
  y_succ BYTE ?

; CHECK: :[[# @LINE + 1]]:7: error: expected non-negative value in struct's 'org' directive; was -4
  ORG -4

  z BYTE ?
  z_succ BYTE ?
org_struct ENDS

; CHECK: :[[# @LINE + 1]]:16: error: cannot initialize a value of type 'org_struct'; 'org' was used in the type's declaration
x org_struct <>

end
