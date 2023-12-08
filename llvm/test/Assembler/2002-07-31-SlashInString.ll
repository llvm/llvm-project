; RUN: llvm-as < %s | llvm-dis | llvm-as -o /dev/null
; RUN: verify-uselistorder %s

; Make sure that \\ works in a string initializer
@Slashtest = internal global [8 x i8] c"\5Cbegin{\00"

