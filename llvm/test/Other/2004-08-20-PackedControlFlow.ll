; RUN: llvm-as < %s | llvm-dis | llvm-as > /dev/null

        %v4f = type <4 x float>
@foo = external global %v4f             ; <%v4f*> [#uses=1]
@bar = external global %v4f             ; <%v4f*> [#uses=1]

define void @main() {
        br label %A

C:              ; preds = %B
        store %v4f %t2, %v4f* @bar
        ret void

B:              ; preds = %A
        %t2 = fadd %v4f %t0, %t0         ; <%v4f> [#uses=1]
        br label %C

A:              ; preds = %0
        %t0 = load %v4f* @foo           ; <%v4f> [#uses=2]
        br label %B
}

