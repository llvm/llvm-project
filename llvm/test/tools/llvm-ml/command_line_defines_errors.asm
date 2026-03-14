; RUN: llvm-ml -filetype=s %s /Fo - /Dtest1=def 2>&1 | FileCheck %s --implicit-check-not=warning:

.code

; CHECK: :[[# @LINE + 1]]:1: warning: redefining 'test1', already defined on the command line
test1 textequ <redef>

end
