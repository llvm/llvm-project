; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

%FunTy = type i32 (i32)

declare i32 @test(i32)   ; Test forward declaration merging

define void @invoke(%FunTy* %x) {
        %foo = call i32 %x( i32 123 )           ; <i32> [#uses=0]
        %foo2 = tail call i32 %x( i32 123 )             ; <i32> [#uses=0]
        ret void
}

define i32 @main(i32 %argc) {
        %retval = call i32 @test( i32 %argc )           ; <i32> [#uses=2]
        %two = add i32 %retval, %retval         ; <i32> [#uses=1]
        %retval2 = invoke i32 @test( i32 %argc )
                        to label %Next unwind label %Error              ; <i32> [#uses=1]

Next:           ; preds = %0
        %two2 = add i32 %two, %retval2          ; <i32> [#uses=1]
        call void @invoke( %FunTy* @test )
        ret i32 %two2

Error:          ; preds = %0
        ret i32 -1
}

define i32 @test(i32 %i0) {
        ret i32 %i0
}
