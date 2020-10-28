; Input used for generating checked-in binaries (trivial.obj.*)
; llc -mtriple=i386-pc-win32 trivial.ll -filetype=obj -o trivial.obj.coff-i386
; llc -mtriple=x86_64-pc-win32 trivial.ll -filetype=obj -o trivial.obj.coff-x86-64

@.str = private unnamed_addr constant [13 x i8] c"Hello World\0A\00", align 1

define i32 @main() nounwind {
entry:
  %call = tail call i32 @puts(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str, i32 0, i32 0)) nounwind
  tail call void bitcast (void (...)* @SomeOtherFunction to void ()*)() nounwind
  ret i32 0
}

declare i32 @puts(i8* nocapture) nounwind

declare void @SomeOtherFunction(...)
