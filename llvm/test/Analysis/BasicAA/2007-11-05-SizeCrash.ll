; RUN: opt < %s -basic-aa -gvn -disable-output
; PR1774

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
        %struct.device = type { [20 x i8] }
        %struct.pci_device_id = type { i32, i32, i32, i32, i32, i32, i64 }
        %struct.usb_bus = type { %struct.device* }
        %struct.usb_hcd = type { %struct.usb_bus, i64, [0 x i64] }
@uhci_pci_ids = constant [1 x %struct.pci_device_id] zeroinitializer

@__mod_pci_device_table = alias [1 x %struct.pci_device_id], [1 x %struct.pci_device_id]* @uhci_pci_ids     
        ; <[1 x %struct.pci_device_id]*> [#uses=0]

define i32 @uhci_suspend(%struct.usb_hcd* %hcd) {
entry:
        %tmp17 = getelementptr %struct.usb_hcd, %struct.usb_hcd* %hcd, i32 0, i32 2, i64 1      
        ; <i64*> [#uses=1]
        %tmp1718 = bitcast i64* %tmp17 to i32*          ; <i32*> [#uses=1]
        %tmp19 = load i32, i32* %tmp1718, align 4            ; <i32> [#uses=0]
        br i1 false, label %cond_true34, label %done_okay

cond_true34:            ; preds = %entry
        %tmp631 = getelementptr %struct.usb_hcd, %struct.usb_hcd* %hcd, i32 0, i32 2, i64
2305843009213693950            ; <i64*> [#uses=1]
        %tmp70 = bitcast i64* %tmp631 to %struct.device**

        %tmp71 = load %struct.device*, %struct.device** %tmp70, align 8

        ret i32 undef

done_okay:              ; preds = %entry
        ret i32 undef
}
