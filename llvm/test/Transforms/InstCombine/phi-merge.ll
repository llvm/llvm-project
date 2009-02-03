; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep {phi i32}
; PR1777

declare i1 @rrr()

define i1 @zxcv() {
entry:
%a = alloca i32
%i = ptrtoint i32* %a to i32
%b = call i1 @rrr()
br i1 %b, label %one, label %two

one:
%x = phi i32 [%i, %entry], [%y, %two]
%c = call i1 @rrr()
br i1 %c, label %two, label %end

two:
%y = phi i32 [%i, %entry], [%x, %one]
%d = call i1 @rrr()
br i1 %d, label %one, label %end

end:
%f = phi i32 [ %x, %one], [%y, %two]
; Change the %f to %i, and the optimizer suddenly becomes a lot smarter
; even though %f must equal %i at this point
%g = inttoptr i32 %f to i32*
store i32 10, i32* %g
%z = call i1 @rrr()
ret i1 %z
}
