; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=future -ppc-asm-full-reg-names < %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-S
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=future -ppc-asm-full-reg-names --filetype=obj < %s | \
; RUN:   llvm-objdump -dr - | FileCheck %s --check-prefix=CHECK-O


; CHECK-S-LABEL: caller
; CHECK-S: b callee@notoc

; CHECK-O-LABEL: caller
; CHECK-O: b
; CHECK-O-NEXT: R_PPC64_REL24_NOTOC callee
define dso_local signext i32 @caller() local_unnamed_addr {
entry:
  %call = tail call signext i32 bitcast (i32 (...)* @callee to i32 ()*)()
  ret i32 %call
}

declare signext i32 @callee(...) local_unnamed_addr


; Some calls can be considered Extrnal Symbols.
; CHECK-S-LABEL: ExternalSymbol
; CHECK-S: b memcpy@notoc

; CHECK-O-LABEL: ExternalSymbol
; CHECK-O: b
; CHECK-O-NEXT: R_PPC64_REL24_NOTOC memcpy
define dso_local void @ExternalSymbol(i8* nocapture %out, i8* nocapture readonly %in, i64 %num) local_unnamed_addr {
entry:
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %out, i8* align 1 %in, i64 %num, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg)


; CHECK-S-LABEL: callerNoTail
; CHECK-S:     bl callee@notoc
; CHECK-S-NOT: nop
; CHECK-S:     bl callee@notoc
; CHECK-S-NOT: nop
; CHECK-S:     blr

; CHECK-O-LABEL: callerNoTail
; CHECK-O:      bl
; CHECK-O-NEXT: R_PPC64_REL24_NOTOC callee
; CHECK-O-NOT:  nop
; CHECK-O:      bl
; CHECK-O-NEXT: R_PPC64_REL24_NOTOC callee
; CHECK-O-NOT:  nop
; CHECK-O:      blr
define dso_local signext i32 @callerNoTail() local_unnamed_addr {
entry:
  %call1 = tail call signext i32 bitcast (i32 (...)* @callee to i32 ()*)()
  %call2 = tail call signext i32 bitcast (i32 (...)* @callee to i32 ()*)()
  %add = add i32 %call1, %call2
  ret i32 %add
}

