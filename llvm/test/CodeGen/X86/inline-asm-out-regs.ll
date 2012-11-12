; RUN: llc < %s -mtriple=i386-unknown-linux-gnu
; PR3391

@pci_indirect = external global { }             ; <{ }*> [#uses=1]
@pcibios_last_bus = external global i32         ; <i32*> [#uses=2]

define void @pci_pcbios_init() nounwind section ".init.text" {
entry:
        br label %bb1.i

bb1.i:          ; preds = %bb6.i.i, %bb1.i, %entry
        %0 = load i32* null, align 8            ; <i32> [#uses=1]
        %1 = icmp ugt i32 %0, 1048575           ; <i1> [#uses=1]
        br i1 %1, label %bb2.i, label %bb1.i

bb2.i:          ; preds = %bb1.i
        %asmtmp.i.i = tail call { i32, i32, i32, i32 } asm "lcall *(%edi); cld\0A\09jc 1f\0A\09xor %ah, %ah\0A1:", "={dx},={ax},={bx},={cx},1,{di},~{dirflag},~{fpsr},~{flags},~{memory}"(i32 45313, { }* @pci_indirect) nounwind             ; <{ i32, i32, i32, i32 }> [#uses=2]
        %asmresult2.i.i = extractvalue { i32, i32, i32, i32 } %asmtmp.i.i, 1   
        ; <i32> [#uses=1]
        %2 = lshr i32 %asmresult2.i.i, 8                ; <i32> [#uses=1]
        %3 = trunc i32 %2 to i8         ; <i8> [#uses=1]
        %4 = load i32* @pcibios_last_bus, align 4               ; <i32> [#uses=1]
        %5 = icmp slt i32 %4, 0         ; <i1> [#uses=1]
        br i1 %5, label %bb5.i.i, label %bb6.i.i

bb5.i.i:                ; preds = %bb2.i
        %asmresult4.i.i = extractvalue { i32, i32, i32, i32 } %asmtmp.i.i, 3   
        ; <i32> [#uses=1]
        %6 = and i32 %asmresult4.i.i, 255               ; <i32> [#uses=1]
        store i32 %6, i32* @pcibios_last_bus, align 4
        br label %bb6.i.i

bb6.i.i:                ; preds = %bb5.i.i, %bb2.i
        %7 = icmp eq i8 %3, 0           ; <i1> [#uses=1]
        %or.cond.i.i = and i1 %7, false         ; <i1> [#uses=1]
        br i1 %or.cond.i.i, label %bb1.i, label %bb8.i.i

bb8.i.i:                ; preds = %bb6.i.i
        unreachable
}
