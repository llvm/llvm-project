; Input used for generating checked-in binaries (trivial.obj.*)
; llc -mtriple=i386-pc-win32 trivial.ll -filetype=obj -o trivial.obj.coff-i386
; llc -mtriple=x86_64-pc-win32 trivial.ll -filetype=obj -o trivial.obj.coff-x86-64

@.str = private unnamed_addr constant [13 x i8] c"Hello World\0A\00", align 1

define i32 @main() nounwind {
entry:
  %call = tail call i32 @puts(ptr @.str) nounwind
  tail call void @SomeOtherFunction() nounwind
  ret i32 0
}

declare i32 @puts(ptr nocapture) nounwind

declare void @SomeOtherFunction(...)
