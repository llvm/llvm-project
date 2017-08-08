; Input used for generating checked-in binaries (trivial.obj.*)
; llc -mtriple=wasm32-unknown-unknown-wasm trivial.ll -filetype=obj -o trivial.obj.wasm

@.str = private unnamed_addr constant [13 x i8] c"Hello World\0A\00", align 1

define i32 @main() nounwind {
entry:
  %call = tail call i32 @puts(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str, i32 0, i32 0)) nounwind
  tail call void bitcast (void (...)* @SomeOtherFunction to void ()*)() nounwind
  ret i32 0
}

declare i32 @puts(i8* nocapture) nounwind

declare void @SomeOtherFunction(...)

@var = global i32 0
@llvm.used = appending global [1 x i8*] [i8* bitcast (i32* @var to i8*)], section "llvm.metadata"
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* null, i8* null }]
