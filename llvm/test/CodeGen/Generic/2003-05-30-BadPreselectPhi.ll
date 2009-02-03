; RUN: llvm-as < %s | llc

;; Date:     May 28, 2003.
;; From:     test/Programs/SingleSource/richards_benchmark.c
;; Function: struct task *handlerfn(struct packet *pkt)
;;
;; Error:    PreSelection puts the arguments of the Phi just before
;;           the Phi instead of in predecessor blocks.  This later
;;           causes llc to produces an invalid register <NULL VALUE>
;;           for the phi arguments.

        %struct..packet = type { %struct..packet*, i32, i32, i32, [4 x i8] }
        %struct..task = type { %struct..task*, i32, i32, %struct..packet*, i32, %struct..task* (%struct..packet*)*, i32, i32 }
@v1 = external global i32               ; <i32*> [#uses=1]
@v2 = external global i32               ; <i32*> [#uses=1]

define %struct..task* @handlerfn(%struct..packet* %pkt.2) {
entry:
        %tmp.1 = icmp ne %struct..packet* %pkt.2, null          ; <i1> [#uses=1]
        br i1 %tmp.1, label %cond_false, label %cond_continue

cond_false:             ; preds = %entry
        br label %cond_continue

cond_continue:          ; preds = %cond_false, %entry
        %mem_tmp.0 = phi i32* [ @v2, %cond_false ], [ @v1, %entry ]             ; <i32*> [#uses=1]
        %tmp.12 = bitcast i32* %mem_tmp.0 to %struct..packet*           ; <%struct..packet*> [#uses=1]
        call void @append( %struct..packet* %pkt.2, %struct..packet* %tmp.12 )
        ret %struct..task* null
}

declare void @append(%struct..packet*, %struct..packet*)

