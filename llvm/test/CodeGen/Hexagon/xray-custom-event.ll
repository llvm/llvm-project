; RUN: llc -mtriple=hexagon-unknown-elf < %s | FileCheck %s
; RUN: llc -mtriple=hexagon-unknown-linux-musl < %s | FileCheck %s

define i32 @customevent(ptr %p, i32 %sz) nounwind noinline uwtable "function-instrument"="xray-always" {
entry:
;; Function entry sled.
; CHECK-LABEL: .Lxray_sled_0:
; CHECK:       jump

;; Custom event sled (12 words = 48 bytes):
;; jump-over, allocframe, sp adjust, 2 saves, 2 moves, call,
;; 2 restores, sp adjust, deallocframe.
; CHECK-LABEL: .Lxray_sled_1:
; CHECK:       jump .Ltmp
; CHECK:       allocframe(r29,#0):raw
; CHECK:       r29 = add(r29,#-8)
; CHECK:       memw(r29+#0) = r0
; CHECK:       memw(r29+#4) = r1
; CHECK:       r0 = r{{[0-9]+}}
; CHECK:       r1 = r{{[0-9]+}}
; CHECK:       call __xray_CustomEvent
; CHECK:       r0 = memw(r29+#0)
; CHECK:       r1 = memw(r29+#4)
; CHECK:       r29 = add(r29,#8)
; CHECK:       deallocframe
; CHECK:       .Ltmp

  call void @llvm.xray.customevent(ptr %p, i64 4)

;; Function exit sled.
; CHECK-LABEL: .Lxray_sled_2:
; CHECK:       jump

  ret i32 0
}

;; Verify the xray_instr_map entry records CUSTOM_EVENT kind (0x04).
; CHECK:       .section xray_instr_map
; CHECK:       .byte 0x04
;; 0x04 = SledKind::CUSTOM_EVENT

define i32 @typedevent(i64 %type, ptr %p, i32 %sz) nounwind noinline uwtable "function-instrument"="xray-always" {
entry:
;; Function entry sled.
; CHECK-LABEL: .Lxray_sled_3:
; CHECK:       jump

;; Typed event sled (15 words = 60 bytes):
;; jump-over, allocframe, sp adjust, 3 saves, 3 moves, call,
;; 3 restores, sp adjust, deallocframe.
; CHECK-LABEL: .Lxray_sled_4:
; CHECK:       jump .Ltmp
; CHECK:       allocframe(r29,#0):raw
; CHECK:       r29 = add(r29,#-12)
; CHECK:       memw(r29+#0) = r0
; CHECK:       memw(r29+#4) = r1
; CHECK:       memw(r29+#8) = r2
; CHECK:       r0 = r{{[0-9]+}}
; CHECK:       r1 = r{{[0-9]+}}
; CHECK:       r2 = r{{[0-9]+}}
; CHECK:       call __xray_TypedEvent
; CHECK:       r0 = memw(r29+#0)
; CHECK:       r1 = memw(r29+#4)
; CHECK:       r2 = memw(r29+#8)
; CHECK:       r29 = add(r29,#12)
; CHECK:       deallocframe
; CHECK:       .Ltmp

  call void @llvm.xray.typedevent(i64 %type, ptr %p, i64 4)

;; Function exit sled.
; CHECK-LABEL: .Lxray_sled_5:
; CHECK:       jump

  ret i32 0
}

;; Verify the xray_instr_map entry records TYPED_EVENT kind (0x05).
; CHECK:       .byte 0x05
;; 0x05 = SledKind::TYPED_EVENT

declare void @llvm.xray.customevent(ptr, i64)
declare void @llvm.xray.typedevent(i64, ptr, i64)
