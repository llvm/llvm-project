; RUN: opt < %s -aa-pipeline=basic-aa -passes=gvn -disable-output
; PR1782

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
	%struct.device = type { [20 x i8] }
	%struct.pci_device_id = type { i32, i32, i32, i32, i32, i32, i64 }
	%struct.usb_bus = type { ptr }
	%struct.usb_hcd = type { %struct.usb_bus, [0 x i64] }
@pci_ids = constant [1 x %struct.pci_device_id] zeroinitializer

@__mod_pci_device_table = alias [1 x %struct.pci_device_id], ptr @pci_ids		; <ptr> [#uses=0]

define i32 @ehci_pci_setup(ptr %hcd) {
entry:
	%tmp14 = getelementptr %struct.usb_hcd, ptr %hcd, i32 0, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp15 = load ptr, ptr %tmp14, align 8		; <ptr> [#uses=0]
	br i1 false, label %bb25, label %return

bb25:		; preds = %entry
	br i1 false, label %cond_true, label %return

cond_true:		; preds = %bb25
	%tmp601 = getelementptr %struct.usb_hcd, ptr %hcd, i32 0, i32 1, i64 2305843009213693951		; <ptr> [#uses=1]
	%tmp68 = load ptr, ptr %tmp601, align 8		; <ptr> [#uses=0]
	ret i32 undef

return:		; preds = %bb25, %entry
	ret i32 undef
}
