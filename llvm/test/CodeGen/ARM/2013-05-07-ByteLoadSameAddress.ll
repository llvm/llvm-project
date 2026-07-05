; RUN: llc -mtriple=thumb-eabi -mattr=+v7,+thumb2 %s -o - | FileCheck %s

define i8 @f1(ptr %call1, ptr %call3, i32 %h, i32 %w, i32 %Width) {
; CHECK: f1:
entry:
        %mul17 = mul nsw i32 %Width, %h
        %add = add nsw i32 %mul17, %w
        %sub19 = sub i32 %add, %Width
        %sub20 = add i32 %sub19, -1
        %arrayidx21 = getelementptr inbounds i8, ptr %call1, i32 %sub20
        %0 = load i8, ptr %arrayidx21, align 1
        %conv22 = zext i8 %0 to i32
        %arrayidx25 = getelementptr inbounds i8, ptr %call1, i32 %sub19
        %1 = load i8, ptr %arrayidx25, align 1
        %conv26 = zext i8 %1 to i32
        %mul23189 = add i32 %conv26, %conv22
        %add30 = add i32 %sub19, 1
        %arrayidx31 = getelementptr inbounds i8, ptr %call1, i32 %add30
        %2 = load i8, ptr %arrayidx31, align 1
        %conv32 = zext i8 %2 to i32
; CHECK: ldrb r{{[0-9]*}}, [r{{[0-9]*}}, #-1]
; CHECK-NEXT: ldrb r{{[0-9]*}}, [r{{[0-9]*}}, #1]
        %add28190 = add i32 %mul23189, %conv32
        %sub35 = add i32 %add, -1
        %arrayidx36 = getelementptr inbounds i8, ptr %call1, i32 %sub35
        %3 = load i8, ptr %arrayidx36, align 1
        %conv37 = zext i8 %3 to i32
        %add34191 = add i32 %add28190, %conv37
        %arrayidx40 = getelementptr inbounds i8, ptr %call1, i32 %add
        %4 = load i8, ptr %arrayidx40, align 1
        %conv41 = zext i8 %4 to i32
        %mul42 = mul nsw i32 %conv41, 255
        %add44 = add i32 %add, 1
        %arrayidx45 = getelementptr inbounds i8, ptr %call1, i32 %add44
        %5 = load i8, ptr %arrayidx45, align 1
        %conv46 = zext i8 %5 to i32
; CHECK: ldrb r{{[0-9]*}}, [r{{[0-9]*}}, #-1]
; CHECK-NEXT: ldrb r{{[0-9]*}}, [r{{[0-9]*}}, #1]
        %add49 = add i32 %add, %Width
        %sub50 = add i32 %add49, -1
        %arrayidx51 = getelementptr inbounds i8, ptr %call1, i32 %sub50
        %6 = load i8, ptr %arrayidx51, align 1
        %conv52 = zext i8 %6 to i32
        %arrayidx56 = getelementptr inbounds i8, ptr %call1, i32 %add49
        %7 = load i8, ptr %arrayidx56, align 1
        %conv57 = zext i8 %7 to i32
        %add61 = add i32 %add49, 1
        %arrayidx62 = getelementptr inbounds i8, ptr %call1, i32 %add61
        %8 = load i8, ptr %arrayidx62, align 1
        %conv63 = zext i8 %8 to i32
; CHECK: ldrb r{{[0-9]*}}, [r{{[0-9]*}}, #-1]
; CHECK-NEXT: ldrb{{[.w]*}} r{{[0-9]*}}, [r{{[0-9]*}}, #1]
        %tmp = add i32 %add34191, %conv46
        %tmp193 = add i32 %tmp, %conv52
        %tmp194 = add i32 %tmp193, %conv57
        %tmp195 = add i32 %tmp194, %conv63
        %tmp196 = mul i32 %tmp195, -28
        %add65 = add i32 %tmp196, %mul42
        %9 = lshr i32 %add65, 8
        %conv68 = trunc i32 %9 to i8
        %arrayidx69 = getelementptr inbounds i8, ptr %call3, i32 %add
        store i8 %conv68, ptr %arrayidx69, align 1
        ret i8 %conv68
}
