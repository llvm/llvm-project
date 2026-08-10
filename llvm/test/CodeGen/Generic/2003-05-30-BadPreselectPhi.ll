; RUN: llc < %s

;; Date:     May 28, 2003.
;; From:     test/Programs/SingleSource/richards_benchmark.c
;; Function: struct task *handlerfn(struct packet *pkt)
;;
;; Error:    PreSelection puts the arguments of the Phi just before
;;           the Phi instead of in predecessor blocks.  This later
;;           causes llc to produces an invalid register <NULL VALUE>
;;           for the phi arguments.

        %struct..packet = type { ptr, i32, i32, i32, [4 x i8] }
        %struct..task = type { ptr, i32, i32, ptr, i32, ptr, i32, i32 }
@v1 = external global i32               ; <ptr> [#uses=1]
@v2 = external global i32               ; <ptr> [#uses=1]

define ptr @handlerfn(ptr %pkt.2) {
entry:
        %tmp.1 = icmp ne ptr %pkt.2, null          ; <i1> [#uses=1]
        br i1 %tmp.1, label %cond_false, label %cond_continue

cond_false:             ; preds = %entry
        br label %cond_continue

cond_continue:          ; preds = %cond_false, %entry
        %mem_tmp.0 = phi ptr [ @v2, %cond_false ], [ @v1, %entry ]             ; <ptr> [#uses=1]
        call void @append( ptr %pkt.2, ptr %mem_tmp.0 )
        ret ptr null
}

declare void @append(ptr, ptr)

