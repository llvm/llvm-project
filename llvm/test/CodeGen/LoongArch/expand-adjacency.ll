; RUN: llc --mtriple=loongarch64 --relocation-model=pic \
; RUN:   --code-model=medium < %s | FileCheck --check-prefix=MEDIUM %s
; RUN: llc --mtriple=loongarch64 --relocation-model=pic \
; RUN:   --code-model=large < %s | FileCheck --check-prefix=LARGE %s
; RUN: llc --mtriple=loongarch64 --relocation-model=pic \
; RUN:   --enable-tlsdesc --code-model=large < %s | \
; RUN:   FileCheck --check-prefix=LARGEDESC %s

; Check the adjancency of pseudo-instruction expansions to ensure
; compliance with psABI requirements:
; https://github.com/loongson/la-abi-specs/releases/tag/v2.30

declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

define void @call_external_sym(ptr %dst) {
; LARGE-LABEL: call_external_sym:
; LARGE:         pcalau12i [[REG1:\$[a-z0-9]+]], %got_pc_hi20(memset)
; LARGE-NEXT:    addi.d [[REG2:\$[a-z0-9]+]], $zero, %got_pc_lo12(memset)
; LARGE-NEXT:    lu32i.d [[REG2]], %got64_pc_lo20(memset)
; LARGE-NEXT:    lu52i.d [[REG2]], [[REG2]], %got64_pc_hi12(memset)
entry:
  call void @llvm.memset.p0.i64(ptr %dst, i8 0, i64 1000, i1 false)
  ret void
}

declare i32 @callee_tail(i32 %i)

define i32 @caller_call_tail(i32 %i) nounwind {
; MEDIUM-LABEL: caller_call_tail:
; MEDIUM:         pcaddu18i $t8, %call36(callee_tail)
; MEDIUM-NEXT:    jr $t8
;
; LARGE-LABEL: caller_call_tail:
; LARGE:         pcalau12i [[REG1:\$[a-z0-9]+]], %got_pc_hi20(callee_tail)
; LARGE-NEXT:    addi.d [[REG2:\$[a-z0-9]+]], $zero, %got_pc_lo12(callee_tail)
; LARGE-NEXT:    lu32i.d [[REG2]], %got64_pc_lo20(callee_tail)
; LARGE-NEXT:    lu52i.d [[REG2]], [[REG2]], %got64_pc_hi12(callee_tail)
entry:
  call i32 @callee_tail(i32 %i)
  %r = tail call i32 @callee_tail(i32 %i)
  ret i32 %r
}

@ie = external thread_local(initialexec) global i32

define void @test_la_tls_ie(i32 signext %n) {
; LARGE-LABEL: test_la_tls_ie:
; LARGE:         pcalau12i [[REG1:\$[a-z0-9]+]], %ie_pc_hi20(ie)
; LARGE-NEXT:    addi.d [[REG2:\$[a-z0-9]+]], $zero, %ie_pc_lo12(ie)
; LARGE-NEXT:    lu32i.d [[REG2]], %ie64_pc_lo20(ie)
; LARGE-NEXT:    lu52i.d [[REG2]], [[REG2]], %ie64_pc_hi12(ie)
entry:
  br label %loop

loop:
  %i = phi i32 [ %inc, %loop ], [ 0, %entry ]
  %0 = load volatile i32, ptr @ie, align 4
  %inc = add nuw nsw i32 %i, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %loop, label %ret

ret:
  ret void
}

@ld = external thread_local(localdynamic) global i32

define void @test_la_tls_ld(i32 signext %n) {
; LARGE-LABEL: test_la_tls_ld:
; LARGE:         pcalau12i [[REG1:\$[a-z0-9]+]], %ld_pc_hi20(ld)
; LARGE-NEXT:    addi.d [[REG2:\$[a-z0-9]+]], $zero, %got_pc_lo12(ld)
; LARGE-NEXT:    lu32i.d [[REG2]], %got64_pc_lo20(ld)
; LARGE-NEXT:    lu52i.d [[REG2]], [[REG2]], %got64_pc_hi12(ld)
entry:
  br label %loop

loop:
  %i = phi i32 [ %inc, %loop ], [ 0, %entry ]
  %0 = load volatile i32, ptr @ld, align 4
  %inc = add nuw nsw i32 %i, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %loop, label %ret

ret:
  ret void
}

@gd = external thread_local global i32

define void @test_la_tls_gd(i32 signext %n) nounwind {
; LARGE-LABEL: test_la_tls_gd:
; LARGE:         pcalau12i [[REG1:\$[a-z0-9]+]], %gd_pc_hi20(gd)
; LARGE-NEXT:    addi.d [[REG2:\$[a-z0-9]+]], $zero, %got_pc_lo12(gd)
; LARGE-NEXT:    lu32i.d [[REG2]], %got64_pc_lo20(gd)
; LARGE-NEXT:    lu52i.d [[REG2]], [[REG2]], %got64_pc_hi12(gd)
entry:
  br label %loop

loop:
  %i = phi i32 [ %inc, %loop ], [ 0, %entry ]
  %0 = load volatile i32, ptr @gd, align 4
  %inc = add nuw nsw i32 %i, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %loop, label %ret

ret:
  ret void
}

@unspecified = external thread_local global i32

define ptr @test_la_tls_desc() nounwind {
; LARGEDESC-LABEL: test_la_tls_desc:
; LARGEDESC:         pcalau12i [[REG1:\$[a-z0-9]+]], %desc_pc_hi20(unspecified)
; LARGEDESC-NEXT:    addi.d [[REG2:\$[a-z0-9]+]], $zero, %desc_pc_lo12(unspecified)
; LARGEDESC-NEXT:    lu32i.d [[REG2]], %desc64_pc_lo20(unspecified)
; LARGEDESC-NEXT:    lu52i.d [[REG2]], [[REG2]], %desc64_pc_hi12(unspecified)
entry:
  ret ptr @unspecified
}
