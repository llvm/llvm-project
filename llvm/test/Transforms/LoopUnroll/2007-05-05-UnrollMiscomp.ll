; RUN: opt < %s -passes=loop-unroll -S | not grep undef
; PR1385

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-apple-darwin8"
        %struct.__mpz_struct = type { i32, i32, i32* }


define void @Foo(%struct.__mpz_struct* %base) {
entry:
        %want = alloca [1 x %struct.__mpz_struct], align 16             ; <[1 x %struct.__mpz_struct]*> [#uses=4]
        %want1 = getelementptr [1 x %struct.__mpz_struct], [1 x %struct.__mpz_struct]* %want, i32 0, i32 0          ; <%struct.__mpz_struct*> [#uses=1]
        call void @__gmpz_init( %struct.__mpz_struct* %want1 )
        %want27 = getelementptr [1 x %struct.__mpz_struct], [1 x %struct.__mpz_struct]* %want, i32 0, i32 0         ; <%struct.__mpz_struct*> [#uses=1]
        %want3 = getelementptr [1 x %struct.__mpz_struct], [1 x %struct.__mpz_struct]* %want, i32 0, i32 0          ; <%struct.__mpz_struct*> [#uses=1]
        %want2 = getelementptr [1 x %struct.__mpz_struct], [1 x %struct.__mpz_struct]* %want, i32 0, i32 0          ; <%struct.__mpz_struct*> [#uses=2]
        br label %bb

bb:             ; preds = %bb, %entry
        %i.01.0 = phi i32 [ 0, %entry ], [ %indvar.next, %bb ]          ; <i32> [#uses=1]
        %want23.0 = phi %struct.__mpz_struct* [ %want27, %entry ], [ %want2, %bb ]              ; <%struct.__mpz_struct*> [#uses=1]
        call void @__gmpz_mul( %struct.__mpz_struct* %want23.0, %struct.__mpz_struct* %want3, %struct.__mpz_struct* %base )
        %indvar.next = add i32 %i.01.0, 1               ; <i32> [#uses=2]
        %exitcond = icmp ne i32 %indvar.next, 2         ; <i1> [#uses=1]
        br i1 %exitcond, label %bb, label %bb10

bb10:           ; preds = %bb
        %want2.lcssa = phi %struct.__mpz_struct* [ %want2, %bb ]                ; <%struct.__mpz_struct*> [#uses=1]
        call void @__gmpz_clear( %struct.__mpz_struct* %want2.lcssa )
        ret void
}

declare void @__gmpz_init(%struct.__mpz_struct*)
declare void @__gmpz_mul(%struct.__mpz_struct*, %struct.__mpz_struct*, %struct.__mpz_struct*)
declare void @__gmpz_clear(%struct.__mpz_struct*)

