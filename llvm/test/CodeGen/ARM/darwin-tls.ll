; RUN: llc -mtriple=thumbv7s-apple-ios7.0 -o - -fast-isel %s | FileCheck %s --check-prefix=T2-MOVT-PIC
; RUN: llc -mtriple=thumbv7s-apple-ios7.0 -o - %s -mattr=+no-movt | FileCheck %s --check-prefix=T2-LIT-PIC
; RUN: llc -mtriple=thumbv7s-apple-ios7.0 -o - %s -relocation-model=static | FileCheck %s --check-prefix=T2-MOVT-STATIC
; RUN: llc -mtriple=thumbv7s-apple-ios7.0 -o - %s -mattr=+no-movt -relocation-model=static | FileCheck %s --check-prefix=T2-LIT-STATIC
; RUN: llc -mtriple=armv7s-apple-ios7.0 -o - %s | FileCheck %s --check-prefix=ARM-MOVT-PIC
; RUN: llc -mtriple=armv7s-apple-ios7.0 -o - %s -mattr=+no-movt | FileCheck %s --check-prefix=ARM-LIT-PIC
; RUN: llc -mtriple=armv7s-apple-ios7.0 -o - %s -relocation-model=static | FileCheck %s --check-prefix=ARM-MOVT-STATIC
; RUN: llc -mtriple=armv7s-apple-ios7.0 -o - %s -mattr=+no-movt -relocation-model=static | FileCheck %s --check-prefix=ARM-LIT-STATIC


@local_tls_var = thread_local global i32 0
@external_tls_var = external thread_local global i32
@hidden_external_tls_var = external hidden thread_local global i32


define i32 @test_local_tls() {
; T2-MOVT-PIC-LABEL: test_local_tls:
; T2-MOVT-PIC: movw r0, :lower16:(_local_tls_var-([[PCREL_LOC:LPC[0-9]+_[0-9]+]]+4))
; T2-MOVT-PIC: movt r0, :upper16:(_local_tls_var-([[PCREL_LOC]]+4))
; T2-MOVT-PIC: [[PCREL_LOC]]:
; T2-MOVT-PIC-NEXT: add r0, pc
; T2-MOVT-PIC: ldr [[TLV_GET_ADDR:r[0-9]+]], [r0]
; T2-MOVT-PIC: blx [[TLV_GET_ADDR]]
; T2-MOVT-PIC: ldr r0, [r0]

; T2-LIT-PIC-LABEL: test_local_tls:
; T2-LIT-PIC: ldr r0, [[LOCAL_VAR_ADDR:LCPI[0-9]+_[0-9]+]]
; T2-LIT-PIC: [[PCREL_LOC:LPC[0-9]+_[0-9]+]]:
; T2-LIT-PIC-NEXT: add r0, pc
; T2-LIT-PIC: ldr [[TLV_GET_ADDR:r[0-9]+]], [r0]
; T2-LIT-PIC: blx [[TLV_GET_ADDR]]
; T2-LIT-PIC: ldr r0, [r0]
; T2-LIT-PIC: [[LOCAL_VAR_ADDR]]:
; T2-LIT-PIC-NEXT: .long _local_tls_var-([[PCREL_LOC]]+4)

; T2-MOVT-STATIC-LABEL: test_local_tls:
; T2-MOVT-STATIC: movw r0, :lower16:_local_tls_var
; T2-MOVT-STATIC: movt r0, :upper16:_local_tls_var
; T2-MOVT-STATIC: ldr [[TLV_GET_ADDR:r[0-9]+]], [r0]
; T2-MOVT-STATIC: blx [[TLV_GET_ADDR]]
; T2-MOVT-STATIC: ldr r0, [r0]

; T2-LIT-STATIC-LABEL: test_local_tls:
; T2-LIT-STATIC: ldr r0, [[LOCAL_VAR_ADDR:LCPI[0-9]+_[0-9]+]]
; T2-LIT-STATIC: ldr [[TLV_GET_ADDR:r[0-9]+]], [r0]
; T2-LIT-STATIC: blx [[TLV_GET_ADDR]]
; T2-LIT-STATIC: ldr r0, [r0]
; T2-LIT-STATIC: [[LOCAL_VAR_ADDR]]:
; T2-LIT-STATIC-NEXT: .long _local_tls_var

; ARM-MOVT-PIC-LABEL: test_local_tls:
; ARM-MOVT-PIC: movw [[VARPC1:r[0-9]+]], :lower16:(_local_tls_var-([[PCREL_LOC1:LPC[0-9]+_[0-9]+]]+8))
; ARM-MOVT-PIC: movt [[VARPC1]], :upper16:(_local_tls_var-([[PCREL_LOC1]]+8))
; ARM-MOVT-PIC: [[PCREL_LOC1]]:
; ARM-MOVT-PIC: add r0, pc, [[VARPC1]]
; ARM-MOVT-PIC: movw [[VARPC2:r[0-9]+]], :lower16:(_local_tls_var-([[PCREL_LOC2:LPC[0-9]+_[0-9]+]]+8))
; ARM-MOVT-PIC: movt [[VARPC2]], :upper16:(_local_tls_var-([[PCREL_LOC2]]+8))
; ARM-MOVT-PIC: [[PCREL_LOC2]]:
; ARM-MOVT-PIC-NEXT: ldr [[TLV_GET_ADDR:r[0-9]+]], [pc, [[VARPC2]]]
; ARM-MOVT-PIC: blx [[TLV_GET_ADDR]]
; ARM-MOVT-PIC: ldr r0, [r0]

; ARM-LIT-PIC-LABEL: test_local_tls:
; ARM-LIT-PIC: ldr r0, [[LOCAL_VAR_ADDR:LCPI[0-9]+_[0-9]+]]
; ARM-LIT-PIC: [[PCREL_LOC:LPC[0-9]+_[0-9]+]]:
; ARM-LIT-PIC-NEXT: add r0, pc
; ARM-LIT-PIC: ldr [[TLV_GET_ADDR:r[0-9]+]], [r0]
; ARM-LIT-PIC: blx [[TLV_GET_ADDR]]
; ARM-LIT-PIC: ldr r0, [r0]
; ARM-LIT-PIC: [[LOCAL_VAR_ADDR]]:
; ARM-LIT-PIC-NEXT: .long _local_tls_var-([[PCREL_LOC]]+8)

; ARM-MOVT-STATIC-LABEL: test_local_tls:
; ARM-MOVT-STATIC: movw r0, :lower16:_local_tls_var
; ARM-MOVT-STATIC: movt r0, :upper16:_local_tls_var
; ARM-MOVT-STATIC: ldr [[TLV_GET_ADDR:r[0-9]+]], [r0]
; ARM-MOVT-STATIC: blx [[TLV_GET_ADDR]]
; ARM-MOVT-STATIC: ldr r0, [r0]

; ARM-LIT-STATIC-LABEL: test_local_tls:
; ARM-LIT-STATIC: ldr r0, [[LOCAL_VAR_ADDR:LCPI[0-9]+_[0-9]+]]
; ARM-LIT-STATIC: ldr [[TLV_GET_ADDR:r[0-9]+]], [r0]
; ARM-LIT-STATIC: blx [[TLV_GET_ADDR]]
; ARM-LIT-STATIC: ldr r0, [r0]
; ARM-LIT-STATIC: [[LOCAL_VAR_ADDR]]:
; ARM-LIT-STATIC-NEXT: .long _local_tls_var


  %val = load i32, ptr @local_tls_var, align 4
  ret i32 %val
}

define i32 @test_external_tls() {
; T2-MOVT-PIC-LABEL: test_external_tls:
; T2-MOVT-PIC: movw r[[EXTGOT:[0-9]+]], :lower16:(L_external_tls_var$non_lazy_ptr-([[PCREL_LOC:LPC[0-9]+_[0-9]+]]+4))
; T2-MOVT-PIC: movt r[[EXTGOT]], :upper16:(L_external_tls_var$non_lazy_ptr-([[PCREL_LOC]]+4))
; T2-MOVT-PIC: [[PCREL_LOC]]:
; T2-MOVT-PIC-NEXT: add r[[EXTGOT]], pc
; T2-MOVT-PIC: ldr r0, [r[[EXTGOT]]]
; T2-MOVT-PIC: ldr [[TLV_GET_ADDR:r[0-9]+]], [r0]
; T2-MOVT-PIC: blx [[TLV_GET_ADDR]]
; T2-MOVT-PIC: ldr r0, [r0]

; T2-LIT-PIC-LABEL: test_external_tls:
; T2-LIT-PIC: ldr r[[EXTGOT:[0-9]+]], [[EXTERNAL_VAR_ADDR:LCPI[0-9]+_[0-9]+]]
; T2-LIT-PIC: [[PCREL_LOC:LPC[0-9]+_[0-9]+]]:
; T2-LIT-PIC-NEXT: add r[[EXTGOT]], pc
; T2-LIT-PIC: ldr r0, [r[[EXTGOT]]]
; T2-LIT-PIC: ldr [[TLV_GET_ADDR:r[0-9]+]], [r0]
; T2-LIT-PIC: blx [[TLV_GET_ADDR]]
; T2-LIT-PIC: ldr r0, [r0]
; T2-LIT-PIC: [[EXTERNAL_VAR_ADDR]]:
; T2-LIT-PIC-NEXT: .long L_external_tls_var$non_lazy_ptr-([[PCREL_LOC]]+4)

; T2-MOVT-STATIC-LABEL: test_external_tls:
; T2-MOVT-STATIC: movw r0, :lower16:_external_tls_var
; T2-MOVT-STATIC: movt r0, :upper16:_external_tls_var
; T2-MOVT-STATIC: ldr [[TLV_GET_ADDR:r[0-9]+]], [r0]
; T2-MOVT-STATIC: blx [[TLV_GET_ADDR]]
; T2-MOVT-STATIC: ldr r0, [r0]

; T2-LIT-STATIC-LABEL: test_external_tls:
; T2-LIT-STATIC: ldr r0, [[EXTERNAL_VAR_ADDR:LCPI[0-9]+_[0-9]+]]
; T2-LIT-STATIC: ldr [[TLV_GET_ADDR:r[0-9]+]], [r0]
; T2-LIT-STATIC: blx [[TLV_GET_ADDR]]
; T2-LIT-STATIC: ldr r0, [r0]
; T2-LIT-STATIC: [[EXTERNAL_VAR_ADDR]]:
; T2-LIT-STATIC-NEXT: .long _external_tls_var

; ARM-MOVT-PIC-LABEL: test_external_tls:
; ARM-MOVT-PIC: movw r[[EXTGOT:[0-9]+]], :lower16:(L_external_tls_var$non_lazy_ptr-([[PCREL_LOC:LPC[0-9]+_[0-9]+]]+8))
; ARM-MOVT-PIC: movt r[[EXTGOT]], :upper16:(L_external_tls_var$non_lazy_ptr-([[PCREL_LOC]]+8))
; ARM-MOVT-PIC: [[PCREL_LOC]]:
; ARM-MOVT-PIC-NEXT: ldr r0, [pc, r[[EXTGOT]]]
; ARM-MOVT-PIC: ldr [[TLV_GET_ADDR:r[0-9]+]], [r0]
; ARM-MOVT-PIC: blx [[TLV_GET_ADDR]]
; ARM-MOVT-PIC: ldr r0, [r0]

; ARM-LIT-PIC-LABEL: test_external_tls:
; ARM-LIT-PIC: ldr r[[EXTGOT:[0-9]+]], [[EXTERNAL_VAR_ADDR:LCPI[0-9]+_[0-9]+]]
; ARM-LIT-PIC: [[PCREL_LOC:LPC[0-9]+_[0-9]+]]:
; ARM-LIT-PIC-NEXT: add r[[EXTGOT]], pc
; ARM-LIT-PIC: ldr r0, [r[[EXTGOT]]]
; ARM-LIT-PIC: ldr [[TLV_GET_ADDR:r[0-9]+]], [r0]
; ARM-LIT-PIC: blx [[TLV_GET_ADDR]]
; ARM-LIT-PIC: ldr r0, [r0]
; ARM-LIT-PIC: [[EXTERNAL_VAR_ADDR]]:
; ARM-LIT-PIC-NEXT: .long L_external_tls_var$non_lazy_ptr-([[PCREL_LOC]]+8)

; ARM-MOVT-STATIC-LABEL: test_external_tls:
; ARM-MOVT-STATIC: movw r0, :lower16:_external_tls_var
; ARM-MOVT-STATIC: movt r0, :upper16:_external_tls_var
; ARM-MOVT-STATIC: ldr [[TLV_GET_ADDR:r[0-9]+]], [r0]
; ARM-MOVT-STATIC: blx [[TLV_GET_ADDR]]
; ARM-MOVT-STATIC: ldr r0, [r0]

; ARM-LIT-STATIC-LABEL: test_external_tls:
; ARM-LIT-STATIC: ldr r0, [[EXTERNAL_VAR_ADDR:LCPI[0-9]+_[0-9]+]]
; ARM-LIT-STATIC: ldr [[TLV_GET_ADDR:r[0-9]+]], [r0]
; ARM-LIT-STATIC: blx [[TLV_GET_ADDR]]
; ARM-LIT-STATIC: ldr r0, [r0]
; ARM-LIT-STATIC: [[EXTERNAL_VAR_ADDR]]:
; ARM-LIT-STATIC-NEXT: .long _external_tls_var

  %val = load i32, ptr @external_tls_var, align 4
  ret i32 %val
}

; Just need something to trigger an indirect reference to the var.
define i32 @use_hidden_external_tls() {
  %val = load i32, ptr @hidden_external_tls_var, align 4
  ret i32 %val
}

; T2-MOVT-PIC: .section __DATA,__thread_ptr,thread_local_variable_pointers
; T2-MOVT-PIC: .p2align 2
; T2-MOVT-PIC: L_external_tls_var$non_lazy_ptr:
; T2-MOVT-PIC:     .indirect_symbol _external_tls_var
; T2-MOVT-PIC:     .long 0
; T2-MOVT-PIC: L_hidden_external_tls_var$non_lazy_ptr:
; T2-MOVT-PIC:     .indirect_symbol _hidden_external_tls_var
; T2-MOVT-PIC:     .long 0
