; RUN: llc < %s -march=x86 >/dev/null
; PR4699

; Handle this extractvalue-of-extractvalue case without getting in
; trouble with CSE in DAGCombine.

        %cc = type { %crd }
        %cr = type { i32 }
        %crd = type { i64, %cr* }
        %pp = type { %cc }

define fastcc void @foo(%pp* nocapture byval %p_arg) {
entry:
        %tmp2 = getelementptr %pp, %pp* %p_arg, i64 0, i32 0         ; <%cc*> [#uses=
        %tmp3 = load %cc, %cc* %tmp2         ; <%cc> [#uses=1]
        %tmp34 = extractvalue %cc %tmp3, 0              ; <%crd> [#uses=1]
        %tmp345 = extractvalue %crd %tmp34, 0           ; <i64> [#uses=1]
        %.ptr.i = load %cr*, %cr** undef              ; <%cr*> [#uses=0]
        %tmp15.i = shl i64 %tmp345, 3           ; <i64> [#uses=0]
        store %cr* undef, %cr** undef
        ret void
}


