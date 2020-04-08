; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=future -ppc-asm-full-reg-names < %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-S
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=future -ppc-asm-full-reg-names --filetype=obj < %s | \
; RUN:   llvm-objdump -dr - | FileCheck %s --check-prefix=CHECK-O


; CHECK-S-LABEL: caller
; CHECK-S: bl callee@notoc
; CHECK-S: blr

; CHECK-O-LABEL: caller
; CHECK-O: bl
; CHECK-O-NEXT: R_PPC64_REL24_NOTOC callee
; CHECK-O: blr
define dso_local signext i32 @caller() local_unnamed_addr {
entry:
  %call = tail call signext i32 bitcast (i32 (...)* @callee to i32 ()*)()
  ret i32 %call
}

declare signext i32 @callee(...) local_unnamed_addr


; Some calls can be considered Extrnal Symbols.
; CHECK-S-LABEL: ExternalSymbol
; CHECK-S: bl memcpy@notoc
; CHECK-S: blr

; CHECK-O-LABEL: ExternalSymbol
; CHECK-O: bl
; CHECK-O-NEXT: R_PPC64_REL24_NOTOC memcpy
; CHECK-O: blr
define dso_local void @ExternalSymbol(i8* nocapture %out, i8* nocapture readonly %in, i64 %num) local_unnamed_addr {
entry:
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %out, i8* align 1 %in, i64 %num, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg)

