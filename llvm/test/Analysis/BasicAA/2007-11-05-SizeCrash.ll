; RUN: opt < %s -aa-pipeline=basic-aa -passes=gvn -disable-output
; PR1774

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
        %struct.device = type { [20 x i8] }
        %struct.pci_device_id = type { i32, i32, i32, i32, i32, i32, i64 }
        %struct.usb_bus = type { ptr }
        %struct.usb_hcd = type { %struct.usb_bus, i64, [0 x i64] }
@uhci_pci_ids = constant [1 x %struct.pci_device_id] zeroinitializer

@__mod_pci_device_table = alias [1 x %struct.pci_device_id], ptr @uhci_pci_ids     
        ; <ptr> [#uses=0]

define i32 @uhci_suspend(ptr %hcd) {
entry:
        %tmp17 = getelementptr %struct.usb_hcd, ptr %hcd, i32 0, i32 2, i64 1      
        ; <ptr> [#uses=1]
        %tmp19 = load i32, ptr %tmp17, align 4            ; <i32> [#uses=0]
        br i1 false, label %cond_true34, label %done_okay

cond_true34:            ; preds = %entry
        %tmp631 = getelementptr %struct.usb_hcd, ptr %hcd, i32 0, i32 2, i64
2305843009213693950            ; <ptr> [#uses=1]

        %tmp71 = load ptr, ptr %tmp631, align 8

        ret i32 undef

done_okay:              ; preds = %entry
        ret i32 undef
}
