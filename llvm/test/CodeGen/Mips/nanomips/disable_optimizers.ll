; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s
; RUIN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs -disable-nm-save-restore < %s | FileCheck %s --check-prefix=CHECK-NO-SAVE-RESTORE
; RUIN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs -disable-nm-lwm-swm < %s | FileCheck %s --check-prefix=CHECK-NO-LWM-SWM
; RUIN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs -disable-nm-pcrel-opt < %s | FileCheck %s --check-prefix=CHECK-NO-PCREL
; RUIN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs -disable-nm-move-opt < %s | FileCheck %s --check-prefix=CHECK-NO-MOVE

; CHECK-LABEL: test4:
; CHECK-NO-SAVE-RESTORE-LABEL: test4:
; CHECK-NO-LWM-SWM-LABEL: test4:
; CHECK-NO-PCREL-LABEL: test4:

define void @test4(i32 %n, ...) {
; CHECK: swm $a1, 4($sp), 7
; CHECK-NO-SAVE-RESTORE: swm $a1, 4($sp), 7
; CHECK-NO-LWM-SWM-NOT: swm
; CHECK-NO-PCREL: swm $a1, 4($sp), 7

  call void asm sideeffect "", ""()
  ret void
}

%struct.bar = type { i32, i32, i32 }

; CHECK-LABEL: square:
; CHECK-NO-SAVE-RESTORE-LABEL: square:
; CHECK-NO-LWM-SWM-LABEL: square:
; CHECK-NO-PCREL-LABEL: square:

define void @square(%struct.bar* %ints) {

; CHECK: lwm $a1, 0($a0), 2
; CHECK-NO-SAVE-RESTORE: lwm $a1, 0($a0), 2
; CHECK-NO-LWM-SWM-NOT: lwm
; CHECK-NO-PCREL: lwm $a1, 0($a0), 2

  %a = getelementptr inbounds %struct.bar, %struct.bar* %ints, i32 0, i32 0
  %1 = load i32, i32* %a, align 4
  %b = getelementptr inbounds %struct.bar, %struct.bar* %ints, i32 0, i32 1
  %2 = load i32, i32* %b, align 4
  %add = add nsw i32 %2, %1
  %c = getelementptr inbounds %struct.bar, %struct.bar* %ints, i32 0, i32 2
  store i32 %add, i32* %c, align 4
  ret void
}

; CHECK-LABEL: test:
; CHECK-NO-SAVE-RESTORE-LABEL: test:
; CHECK-NO-LWM-SWM-LABEL: test:
; CHECK-NO-PCREL-LABEL: test:

; Make sure that SAVE/RESTORE instructions are used for saving and restoring callee-saved registers.
define void @test() {
; CHECK: save 32, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7
; CHECK-NO-SAVE-RESTORE-NOT: save
; CHECK-NO-LWM-SWM: save 32, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7
; CHECK-NO-PCREL: save 32, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7

  call void asm sideeffect "", "~{$16},~{$17},~{$18},~{$19},~{$20},~{$21},~{$23},~{$22},~{$1}"() ret void

; CHECK: restore.jrc 32, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7
; CHECK-NO-SAVE-RESTORE-NOT: restore
; CHECK-NO-LWM-SWM: restore.jrc 32, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7
; CHECK-NO-PCREL: restore.jrc 32, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7

}


@a = external dso_local local_unnamed_addr global [10 x i32], align 4 
@b = external dso_local local_unnamed_addr global [10 x i32], align 4 


; CHECK-LABEL: test_pcrel:
; CHECK-NO-SAVE-RESTORE-LABEL: test_pcrel:
; CHECK-NO-LWM-SWM-LABEL: test_pcrel:
; CHECK-NO-PCREL-LABEL: test_pcrel:

define i32 @test_pcrel() {

; CHECK: lwpc {{.*}}, a+8
; CHECK-NO-SAVE-RESTORE: lwpc {{.*}}, a+8
; CHECK-NO-LWM-SWM: lwpc {{.*}}, a+8
; CHECK-NO-PCREL-NOT: lwpc
  %l = load i32, i32* getelementptr inbounds ([10 x i32], [10 x i32]* @a, i32 0, i32 2), align 4

; CHECK: swpc {{.*}}, b+32
; CHECK-NO-SAVE-RESTORE: swpc {{.*}}, b+32
; CHECK-NO-LWM-SWM: swpc {{.*}}, b+32
; CHECK-NO-PCREL-NOT: swpc
  store i32 %l, i32* getelementptr inbounds ([10 x i32], [10 x i32]* @b, i32 0, i32 8), align 4
  ret i32 %l
}


; Move optimiser disable

declare i32 @bar(i32, i32)

; CHECK-NO-SAVE-RESTORE-LABEL: movep:
; CHECK-NO-LWM-SWM-LABEL: movep:
; CHECK-NO-PCREL-LABEL: movep:

; CHECK-LABEL: movep:
; CHECK-NO-PCREL-LABEL: movep:

define void @movep(i32 %a, i32 %b, i32 %c, i32 %d) {
; CHECK-NO-MOVE-NOT: movep
; CHECK: movep $s1, $s0, $a0, $a1
; CHECK: movep $a0, $a1, $a2, $a3
  call i32 @bar(i32 %c, i32 %d)
; CHECK: movep $a0, $a1, $s1, $s0
  call i32 @bar(i32 %a, i32 %b)
  ret void
}

