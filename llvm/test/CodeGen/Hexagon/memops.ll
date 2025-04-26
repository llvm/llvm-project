; RUN: llc -mtriple=hexagon -mcpu=hexagonv5 < %s | FileCheck %s
; Generate MemOps for V4 and above.

define void @memop_unsigned_char_add5(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_char_add5:
; CHECK:  memb(r{{[0-9]+}}+#0) += #5
  %0 = load i8, ptr %p, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 5
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %p, align 1
  ret void
}

define void @memop_unsigned_char_add(ptr nocapture %p, i8 zeroext %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_char_add:
; CHECK:  memb(r{{[0-9]+}}+#0) += r{{[0-9]+}}
  %conv = zext i8 %x to i32
  %0 = load i8, ptr %p, align 1
  %conv1 = zext i8 %0 to i32
  %add = add nsw i32 %conv1, %conv
  %conv2 = trunc i32 %add to i8
  store i8 %conv2, ptr %p, align 1
  ret void
}

define void @memop_unsigned_char_sub(ptr nocapture %p, i8 zeroext %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_char_sub:
; CHECK:  memb(r{{[0-9]+}}+#0) -= r{{[0-9]+}}
  %conv = zext i8 %x to i32
  %0 = load i8, ptr %p, align 1
  %conv1 = zext i8 %0 to i32
  %sub = sub nsw i32 %conv1, %conv
  %conv2 = trunc i32 %sub to i8
  store i8 %conv2, ptr %p, align 1
  ret void
}

define void @memop_unsigned_char_or(ptr nocapture %p, i8 zeroext %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_char_or:
; CHECK:  memb(r{{[0-9]+}}+#0) |= r{{[0-9]+}}
  %0 = load i8, ptr %p, align 1
  %or3 = or i8 %0, %x
  store i8 %or3, ptr %p, align 1
  ret void
}

define void @memop_unsigned_char_and(ptr nocapture %p, i8 zeroext %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_char_and:
; CHECK:  memb(r{{[0-9]+}}+#0) &= r{{[0-9]+}}
  %0 = load i8, ptr %p, align 1
  %and3 = and i8 %0, %x
  store i8 %and3, ptr %p, align 1
  ret void
}

define void @memop_unsigned_char_clrbit(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_char_clrbit:
; CHECK:  memb(r{{[0-9]+}}+#0) = clrbit(#5)
  %0 = load i8, ptr %p, align 1
  %conv = zext i8 %0 to i32
  %and = and i32 %conv, 223
  %conv1 = trunc i32 %and to i8
  store i8 %conv1, ptr %p, align 1
  ret void
}

define void @memop_unsigned_char_setbit(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_char_setbit:
; CHECK:  memb(r{{[0-9]+}}+#0) = setbit(#7)
  %0 = load i8, ptr %p, align 1
  %conv = zext i8 %0 to i32
  %or = or i32 %conv, 128
  %conv1 = trunc i32 %or to i8
  store i8 %conv1, ptr %p, align 1
  ret void
}

define void @memop_unsigned_char_add5_index(ptr nocapture %p, i32 %i) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_char_add5_index:
; CHECK:  memb(r{{[0-9]+}}+#0) += #5
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 %i
  %0 = load i8, ptr %add.ptr, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 5
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %add.ptr, align 1
  ret void
}

define void @memop_unsigned_char_add_index(ptr nocapture %p, i32 %i, i8 zeroext %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_char_add_index:
; CHECK:  memb(r{{[0-9]+}}+#0) += r{{[0-9]+}}
  %conv = zext i8 %x to i32
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 %i
  %0 = load i8, ptr %add.ptr, align 1
  %conv1 = zext i8 %0 to i32
  %add = add nsw i32 %conv1, %conv
  %conv2 = trunc i32 %add to i8
  store i8 %conv2, ptr %add.ptr, align 1
  ret void
}

define void @memop_unsigned_char_sub_index(ptr nocapture %p, i32 %i, i8 zeroext %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_char_sub_index:
; CHECK:  memb(r{{[0-9]+}}+#0) -= r{{[0-9]+}}
  %conv = zext i8 %x to i32
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 %i
  %0 = load i8, ptr %add.ptr, align 1
  %conv1 = zext i8 %0 to i32
  %sub = sub nsw i32 %conv1, %conv
  %conv2 = trunc i32 %sub to i8
  store i8 %conv2, ptr %add.ptr, align 1
  ret void
}

define void @memop_unsigned_char_or_index(ptr nocapture %p, i32 %i, i8 zeroext %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_char_or_index:
; CHECK:  memb(r{{[0-9]+}}+#0) |= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 %i
  %0 = load i8, ptr %add.ptr, align 1
  %or3 = or i8 %0, %x
  store i8 %or3, ptr %add.ptr, align 1
  ret void
}

define void @memop_unsigned_char_and_index(ptr nocapture %p, i32 %i, i8 zeroext %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_char_and_index:
; CHECK:  memb(r{{[0-9]+}}+#0) &= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 %i
  %0 = load i8, ptr %add.ptr, align 1
  %and3 = and i8 %0, %x
  store i8 %and3, ptr %add.ptr, align 1
  ret void
}

define void @memop_unsigned_char_clrbit_index(ptr nocapture %p, i32 %i) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_char_clrbit_index:
; CHECK:  memb(r{{[0-9]+}}+#0) = clrbit(#5)
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 %i
  %0 = load i8, ptr %add.ptr, align 1
  %conv = zext i8 %0 to i32
  %and = and i32 %conv, 223
  %conv1 = trunc i32 %and to i8
  store i8 %conv1, ptr %add.ptr, align 1
  ret void
}

define void @memop_unsigned_char_setbit_index(ptr nocapture %p, i32 %i) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_char_setbit_index:
; CHECK:  memb(r{{[0-9]+}}+#0) = setbit(#7)
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 %i
  %0 = load i8, ptr %add.ptr, align 1
  %conv = zext i8 %0 to i32
  %or = or i32 %conv, 128
  %conv1 = trunc i32 %or to i8
  store i8 %conv1, ptr %add.ptr, align 1
  ret void
}

define void @memop_unsigned_char_add5_index5(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_char_add5_index5:
; CHECK:  memb(r{{[0-9]+}}+#5) += #5
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 5
  %0 = load i8, ptr %add.ptr, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 5
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %add.ptr, align 1
  ret void
}

define void @memop_unsigned_char_add_index5(ptr nocapture %p, i8 zeroext %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_char_add_index5:
; CHECK:  memb(r{{[0-9]+}}+#5) += r{{[0-9]+}}
  %conv = zext i8 %x to i32
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 5
  %0 = load i8, ptr %add.ptr, align 1
  %conv1 = zext i8 %0 to i32
  %add = add nsw i32 %conv1, %conv
  %conv2 = trunc i32 %add to i8
  store i8 %conv2, ptr %add.ptr, align 1
  ret void
}

define void @memop_unsigned_char_sub_index5(ptr nocapture %p, i8 zeroext %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_char_sub_index5:
; CHECK:  memb(r{{[0-9]+}}+#5) -= r{{[0-9]+}}
  %conv = zext i8 %x to i32
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 5
  %0 = load i8, ptr %add.ptr, align 1
  %conv1 = zext i8 %0 to i32
  %sub = sub nsw i32 %conv1, %conv
  %conv2 = trunc i32 %sub to i8
  store i8 %conv2, ptr %add.ptr, align 1
  ret void
}

define void @memop_unsigned_char_or_index5(ptr nocapture %p, i8 zeroext %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_char_or_index5:
; CHECK:  memb(r{{[0-9]+}}+#5) |= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 5
  %0 = load i8, ptr %add.ptr, align 1
  %or3 = or i8 %0, %x
  store i8 %or3, ptr %add.ptr, align 1
  ret void
}

define void @memop_unsigned_char_and_index5(ptr nocapture %p, i8 zeroext %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_char_and_index5:
; CHECK:  memb(r{{[0-9]+}}+#5) &= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 5
  %0 = load i8, ptr %add.ptr, align 1
  %and3 = and i8 %0, %x
  store i8 %and3, ptr %add.ptr, align 1
  ret void
}

define void @memop_unsigned_char_clrbit_index5(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_char_clrbit_index5:
; CHECK:  memb(r{{[0-9]+}}+#5) = clrbit(#5)
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 5
  %0 = load i8, ptr %add.ptr, align 1
  %conv = zext i8 %0 to i32
  %and = and i32 %conv, 223
  %conv1 = trunc i32 %and to i8
  store i8 %conv1, ptr %add.ptr, align 1
  ret void
}

define void @memop_unsigned_char_setbit_index5(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_char_setbit_index5:
; CHECK:  memb(r{{[0-9]+}}+#5) = setbit(#7)
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 5
  %0 = load i8, ptr %add.ptr, align 1
  %conv = zext i8 %0 to i32
  %or = or i32 %conv, 128
  %conv1 = trunc i32 %or to i8
  store i8 %conv1, ptr %add.ptr, align 1
  ret void
}

define void @memop_signed_char_add5(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_signed_char_add5:
; CHECK:  memb(r{{[0-9]+}}+#0) += #5
  %0 = load i8, ptr %p, align 1
  %conv2 = zext i8 %0 to i32
  %add = add nsw i32 %conv2, 5
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %p, align 1
  ret void
}

define void @memop_signed_char_add(ptr nocapture %p, i8 signext %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_char_add:
; CHECK:  memb(r{{[0-9]+}}+#0) += r{{[0-9]+}}
  %conv4 = zext i8 %x to i32
  %0 = load i8, ptr %p, align 1
  %conv13 = zext i8 %0 to i32
  %add = add nsw i32 %conv13, %conv4
  %conv2 = trunc i32 %add to i8
  store i8 %conv2, ptr %p, align 1
  ret void
}

define void @memop_signed_char_sub(ptr nocapture %p, i8 signext %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_char_sub:
; CHECK:  memb(r{{[0-9]+}}+#0) -= r{{[0-9]+}}
  %conv4 = zext i8 %x to i32
  %0 = load i8, ptr %p, align 1
  %conv13 = zext i8 %0 to i32
  %sub = sub nsw i32 %conv13, %conv4
  %conv2 = trunc i32 %sub to i8
  store i8 %conv2, ptr %p, align 1
  ret void
}

define void @memop_signed_char_or(ptr nocapture %p, i8 signext %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_char_or:
; CHECK:  memb(r{{[0-9]+}}+#0) |= r{{[0-9]+}}
  %0 = load i8, ptr %p, align 1
  %or3 = or i8 %0, %x
  store i8 %or3, ptr %p, align 1
  ret void
}

define void @memop_signed_char_and(ptr nocapture %p, i8 signext %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_char_and:
; CHECK:  memb(r{{[0-9]+}}+#0) &= r{{[0-9]+}}
  %0 = load i8, ptr %p, align 1
  %and3 = and i8 %0, %x
  store i8 %and3, ptr %p, align 1
  ret void
}

define void @memop_signed_char_clrbit(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_signed_char_clrbit:
; CHECK:  memb(r{{[0-9]+}}+#0) = clrbit(#5)
  %0 = load i8, ptr %p, align 1
  %conv2 = zext i8 %0 to i32
  %and = and i32 %conv2, 223
  %conv1 = trunc i32 %and to i8
  store i8 %conv1, ptr %p, align 1
  ret void
}

define void @memop_signed_char_setbit(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_signed_char_setbit:
; CHECK:  memb(r{{[0-9]+}}+#0) = setbit(#7)
  %0 = load i8, ptr %p, align 1
  %conv2 = zext i8 %0 to i32
  %or = or i32 %conv2, 128
  %conv1 = trunc i32 %or to i8
  store i8 %conv1, ptr %p, align 1
  ret void
}

define void @memop_signed_char_add5_index(ptr nocapture %p, i32 %i) nounwind {
entry:
; CHECK-LABEL: memop_signed_char_add5_index:
; CHECK:  memb(r{{[0-9]+}}+#0) += #5
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 %i
  %0 = load i8, ptr %add.ptr, align 1
  %conv2 = zext i8 %0 to i32
  %add = add nsw i32 %conv2, 5
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %add.ptr, align 1
  ret void
}

define void @memop_signed_char_add_index(ptr nocapture %p, i32 %i, i8 signext %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_char_add_index:
; CHECK:  memb(r{{[0-9]+}}+#0) += r{{[0-9]+}}
  %conv4 = zext i8 %x to i32
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 %i
  %0 = load i8, ptr %add.ptr, align 1
  %conv13 = zext i8 %0 to i32
  %add = add nsw i32 %conv13, %conv4
  %conv2 = trunc i32 %add to i8
  store i8 %conv2, ptr %add.ptr, align 1
  ret void
}

define void @memop_signed_char_sub_index(ptr nocapture %p, i32 %i, i8 signext %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_char_sub_index:
; CHECK:  memb(r{{[0-9]+}}+#0) -= r{{[0-9]+}}
  %conv4 = zext i8 %x to i32
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 %i
  %0 = load i8, ptr %add.ptr, align 1
  %conv13 = zext i8 %0 to i32
  %sub = sub nsw i32 %conv13, %conv4
  %conv2 = trunc i32 %sub to i8
  store i8 %conv2, ptr %add.ptr, align 1
  ret void
}

define void @memop_signed_char_or_index(ptr nocapture %p, i32 %i, i8 signext %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_char_or_index:
; CHECK:  memb(r{{[0-9]+}}+#0) |= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 %i
  %0 = load i8, ptr %add.ptr, align 1
  %or3 = or i8 %0, %x
  store i8 %or3, ptr %add.ptr, align 1
  ret void
}

define void @memop_signed_char_and_index(ptr nocapture %p, i32 %i, i8 signext %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_char_and_index:
; CHECK:  memb(r{{[0-9]+}}+#0) &= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 %i
  %0 = load i8, ptr %add.ptr, align 1
  %and3 = and i8 %0, %x
  store i8 %and3, ptr %add.ptr, align 1
  ret void
}

define void @memop_signed_char_clrbit_index(ptr nocapture %p, i32 %i) nounwind {
entry:
; CHECK-LABEL: memop_signed_char_clrbit_index:
; CHECK:  memb(r{{[0-9]+}}+#0) = clrbit(#5)
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 %i
  %0 = load i8, ptr %add.ptr, align 1
  %conv2 = zext i8 %0 to i32
  %and = and i32 %conv2, 223
  %conv1 = trunc i32 %and to i8
  store i8 %conv1, ptr %add.ptr, align 1
  ret void
}

define void @memop_signed_char_setbit_index(ptr nocapture %p, i32 %i) nounwind {
entry:
; CHECK-LABEL: memop_signed_char_setbit_index:
; CHECK:  memb(r{{[0-9]+}}+#0) = setbit(#7)
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 %i
  %0 = load i8, ptr %add.ptr, align 1
  %conv2 = zext i8 %0 to i32
  %or = or i32 %conv2, 128
  %conv1 = trunc i32 %or to i8
  store i8 %conv1, ptr %add.ptr, align 1
  ret void
}

define void @memop_signed_char_add5_index5(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_signed_char_add5_index5:
; CHECK:  memb(r{{[0-9]+}}+#5) += #5
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 5
  %0 = load i8, ptr %add.ptr, align 1
  %conv2 = zext i8 %0 to i32
  %add = add nsw i32 %conv2, 5
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %add.ptr, align 1
  ret void
}

define void @memop_signed_char_add_index5(ptr nocapture %p, i8 signext %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_char_add_index5:
; CHECK:  memb(r{{[0-9]+}}+#5) += r{{[0-9]+}}
  %conv4 = zext i8 %x to i32
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 5
  %0 = load i8, ptr %add.ptr, align 1
  %conv13 = zext i8 %0 to i32
  %add = add nsw i32 %conv13, %conv4
  %conv2 = trunc i32 %add to i8
  store i8 %conv2, ptr %add.ptr, align 1
  ret void
}

define void @memop_signed_char_sub_index5(ptr nocapture %p, i8 signext %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_char_sub_index5:
; CHECK:  memb(r{{[0-9]+}}+#5) -= r{{[0-9]+}}
  %conv4 = zext i8 %x to i32
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 5
  %0 = load i8, ptr %add.ptr, align 1
  %conv13 = zext i8 %0 to i32
  %sub = sub nsw i32 %conv13, %conv4
  %conv2 = trunc i32 %sub to i8
  store i8 %conv2, ptr %add.ptr, align 1
  ret void
}

define void @memop_signed_char_or_index5(ptr nocapture %p, i8 signext %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_char_or_index5:
; CHECK:  memb(r{{[0-9]+}}+#5) |= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 5
  %0 = load i8, ptr %add.ptr, align 1
  %or3 = or i8 %0, %x
  store i8 %or3, ptr %add.ptr, align 1
  ret void
}

define void @memop_signed_char_and_index5(ptr nocapture %p, i8 signext %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_char_and_index5:
; CHECK:  memb(r{{[0-9]+}}+#5) &= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 5
  %0 = load i8, ptr %add.ptr, align 1
  %and3 = and i8 %0, %x
  store i8 %and3, ptr %add.ptr, align 1
  ret void
}

define void @memop_signed_char_clrbit_index5(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_signed_char_clrbit_index5:
; CHECK:  memb(r{{[0-9]+}}+#5) = clrbit(#5)
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 5
  %0 = load i8, ptr %add.ptr, align 1
  %conv2 = zext i8 %0 to i32
  %and = and i32 %conv2, 223
  %conv1 = trunc i32 %and to i8
  store i8 %conv1, ptr %add.ptr, align 1
  ret void
}

define void @memop_signed_char_setbit_index5(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_signed_char_setbit_index5:
; CHECK:  memb(r{{[0-9]+}}+#5) = setbit(#7)
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 5
  %0 = load i8, ptr %add.ptr, align 1
  %conv2 = zext i8 %0 to i32
  %or = or i32 %conv2, 128
  %conv1 = trunc i32 %or to i8
  store i8 %conv1, ptr %add.ptr, align 1
  ret void
}

define void @memop_unsigned_short_add5(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_short_add5:
; CHECK:  memh(r{{[0-9]+}}+#0) += #5
  %0 = load i16, ptr %p, align 2
  %conv = zext i16 %0 to i32
  %add = add nsw i32 %conv, 5
  %conv1 = trunc i32 %add to i16
  store i16 %conv1, ptr %p, align 2
  ret void
}

define void @memop_unsigned_short_add(ptr nocapture %p, i16 zeroext %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_short_add:
; CHECK:  memh(r{{[0-9]+}}+#0) += r{{[0-9]+}}
  %conv = zext i16 %x to i32
  %0 = load i16, ptr %p, align 2
  %conv1 = zext i16 %0 to i32
  %add = add nsw i32 %conv1, %conv
  %conv2 = trunc i32 %add to i16
  store i16 %conv2, ptr %p, align 2
  ret void
}

define void @memop_unsigned_short_sub(ptr nocapture %p, i16 zeroext %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_short_sub:
; CHECK:  memh(r{{[0-9]+}}+#0) -= r{{[0-9]+}}
  %conv = zext i16 %x to i32
  %0 = load i16, ptr %p, align 2
  %conv1 = zext i16 %0 to i32
  %sub = sub nsw i32 %conv1, %conv
  %conv2 = trunc i32 %sub to i16
  store i16 %conv2, ptr %p, align 2
  ret void
}

define void @memop_unsigned_short_or(ptr nocapture %p, i16 zeroext %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_short_or:
; CHECK:  memh(r{{[0-9]+}}+#0) |= r{{[0-9]+}}
  %0 = load i16, ptr %p, align 2
  %or3 = or i16 %0, %x
  store i16 %or3, ptr %p, align 2
  ret void
}

define void @memop_unsigned_short_and(ptr nocapture %p, i16 zeroext %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_short_and:
; CHECK:  memh(r{{[0-9]+}}+#0) &= r{{[0-9]+}}
  %0 = load i16, ptr %p, align 2
  %and3 = and i16 %0, %x
  store i16 %and3, ptr %p, align 2
  ret void
}

define void @memop_unsigned_short_clrbit(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_short_clrbit:
; CHECK:  memh(r{{[0-9]+}}+#0) = clrbit(#5)
  %0 = load i16, ptr %p, align 2
  %conv = zext i16 %0 to i32
  %and = and i32 %conv, 65503
  %conv1 = trunc i32 %and to i16
  store i16 %conv1, ptr %p, align 2
  ret void
}

define void @memop_unsigned_short_setbit(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_short_setbit:
; CHECK:  memh(r{{[0-9]+}}+#0) = setbit(#7)
  %0 = load i16, ptr %p, align 2
  %conv = zext i16 %0 to i32
  %or = or i32 %conv, 128
  %conv1 = trunc i32 %or to i16
  store i16 %conv1, ptr %p, align 2
  ret void
}

define void @memop_unsigned_short_add5_index(ptr nocapture %p, i32 %i) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_short_add5_index:
; CHECK:  memh(r{{[0-9]+}}+#0) += #5
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 %i
  %0 = load i16, ptr %add.ptr, align 2
  %conv = zext i16 %0 to i32
  %add = add nsw i32 %conv, 5
  %conv1 = trunc i32 %add to i16
  store i16 %conv1, ptr %add.ptr, align 2
  ret void
}

define void @memop_unsigned_short_add_index(ptr nocapture %p, i32 %i, i16 zeroext %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_short_add_index:
; CHECK:  memh(r{{[0-9]+}}+#0) += r{{[0-9]+}}
  %conv = zext i16 %x to i32
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 %i
  %0 = load i16, ptr %add.ptr, align 2
  %conv1 = zext i16 %0 to i32
  %add = add nsw i32 %conv1, %conv
  %conv2 = trunc i32 %add to i16
  store i16 %conv2, ptr %add.ptr, align 2
  ret void
}

define void @memop_unsigned_short_sub_index(ptr nocapture %p, i32 %i, i16 zeroext %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_short_sub_index:
; CHECK:  memh(r{{[0-9]+}}+#0) -= r{{[0-9]+}}
  %conv = zext i16 %x to i32
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 %i
  %0 = load i16, ptr %add.ptr, align 2
  %conv1 = zext i16 %0 to i32
  %sub = sub nsw i32 %conv1, %conv
  %conv2 = trunc i32 %sub to i16
  store i16 %conv2, ptr %add.ptr, align 2
  ret void
}

define void @memop_unsigned_short_or_index(ptr nocapture %p, i32 %i, i16 zeroext %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_short_or_index:
; CHECK:  memh(r{{[0-9]+}}+#0) |= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 %i
  %0 = load i16, ptr %add.ptr, align 2
  %or3 = or i16 %0, %x
  store i16 %or3, ptr %add.ptr, align 2
  ret void
}

define void @memop_unsigned_short_and_index(ptr nocapture %p, i32 %i, i16 zeroext %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_short_and_index:
; CHECK:  memh(r{{[0-9]+}}+#0) &= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 %i
  %0 = load i16, ptr %add.ptr, align 2
  %and3 = and i16 %0, %x
  store i16 %and3, ptr %add.ptr, align 2
  ret void
}

define void @memop_unsigned_short_clrbit_index(ptr nocapture %p, i32 %i) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_short_clrbit_index:
; CHECK:  memh(r{{[0-9]+}}+#0) = clrbit(#5)
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 %i
  %0 = load i16, ptr %add.ptr, align 2
  %conv = zext i16 %0 to i32
  %and = and i32 %conv, 65503
  %conv1 = trunc i32 %and to i16
  store i16 %conv1, ptr %add.ptr, align 2
  ret void
}

define void @memop_unsigned_short_setbit_index(ptr nocapture %p, i32 %i) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_short_setbit_index:
; CHECK:  memh(r{{[0-9]+}}+#0) = setbit(#7)
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 %i
  %0 = load i16, ptr %add.ptr, align 2
  %conv = zext i16 %0 to i32
  %or = or i32 %conv, 128
  %conv1 = trunc i32 %or to i16
  store i16 %conv1, ptr %add.ptr, align 2
  ret void
}

define void @memop_unsigned_short_add5_index5(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_short_add5_index5:
; CHECK:  memh(r{{[0-9]+}}+#10) += #5
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 5
  %0 = load i16, ptr %add.ptr, align 2
  %conv = zext i16 %0 to i32
  %add = add nsw i32 %conv, 5
  %conv1 = trunc i32 %add to i16
  store i16 %conv1, ptr %add.ptr, align 2
  ret void
}

define void @memop_unsigned_short_add_index5(ptr nocapture %p, i16 zeroext %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_short_add_index5:
; CHECK:  memh(r{{[0-9]+}}+#10) += r{{[0-9]+}}
  %conv = zext i16 %x to i32
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 5
  %0 = load i16, ptr %add.ptr, align 2
  %conv1 = zext i16 %0 to i32
  %add = add nsw i32 %conv1, %conv
  %conv2 = trunc i32 %add to i16
  store i16 %conv2, ptr %add.ptr, align 2
  ret void
}

define void @memop_unsigned_short_sub_index5(ptr nocapture %p, i16 zeroext %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_short_sub_index5:
; CHECK:  memh(r{{[0-9]+}}+#10) -= r{{[0-9]+}}
  %conv = zext i16 %x to i32
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 5
  %0 = load i16, ptr %add.ptr, align 2
  %conv1 = zext i16 %0 to i32
  %sub = sub nsw i32 %conv1, %conv
  %conv2 = trunc i32 %sub to i16
  store i16 %conv2, ptr %add.ptr, align 2
  ret void
}

define void @memop_unsigned_short_or_index5(ptr nocapture %p, i16 zeroext %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_short_or_index5:
; CHECK:  memh(r{{[0-9]+}}+#10) |= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 5
  %0 = load i16, ptr %add.ptr, align 2
  %or3 = or i16 %0, %x
  store i16 %or3, ptr %add.ptr, align 2
  ret void
}

define void @memop_unsigned_short_and_index5(ptr nocapture %p, i16 zeroext %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_short_and_index5:
; CHECK:  memh(r{{[0-9]+}}+#10) &= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 5
  %0 = load i16, ptr %add.ptr, align 2
  %and3 = and i16 %0, %x
  store i16 %and3, ptr %add.ptr, align 2
  ret void
}

define void @memop_unsigned_short_clrbit_index5(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_short_clrbit_index5:
; CHECK:  memh(r{{[0-9]+}}+#10) = clrbit(#5)
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 5
  %0 = load i16, ptr %add.ptr, align 2
  %conv = zext i16 %0 to i32
  %and = and i32 %conv, 65503
  %conv1 = trunc i32 %and to i16
  store i16 %conv1, ptr %add.ptr, align 2
  ret void
}

define void @memop_unsigned_short_setbit_index5(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_short_setbit_index5:
; CHECK:  memh(r{{[0-9]+}}+#10) = setbit(#7)
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 5
  %0 = load i16, ptr %add.ptr, align 2
  %conv = zext i16 %0 to i32
  %or = or i32 %conv, 128
  %conv1 = trunc i32 %or to i16
  store i16 %conv1, ptr %add.ptr, align 2
  ret void
}

define void @memop_signed_short_add5(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_signed_short_add5:
; CHECK:  memh(r{{[0-9]+}}+#0) += #5
  %0 = load i16, ptr %p, align 2
  %conv2 = zext i16 %0 to i32
  %add = add nsw i32 %conv2, 5
  %conv1 = trunc i32 %add to i16
  store i16 %conv1, ptr %p, align 2
  ret void
}

define void @memop_signed_short_add(ptr nocapture %p, i16 signext %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_short_add:
; CHECK:  memh(r{{[0-9]+}}+#0) += r{{[0-9]+}}
  %conv4 = zext i16 %x to i32
  %0 = load i16, ptr %p, align 2
  %conv13 = zext i16 %0 to i32
  %add = add nsw i32 %conv13, %conv4
  %conv2 = trunc i32 %add to i16
  store i16 %conv2, ptr %p, align 2
  ret void
}

define void @memop_signed_short_sub(ptr nocapture %p, i16 signext %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_short_sub:
; CHECK:  memh(r{{[0-9]+}}+#0) -= r{{[0-9]+}}
  %conv4 = zext i16 %x to i32
  %0 = load i16, ptr %p, align 2
  %conv13 = zext i16 %0 to i32
  %sub = sub nsw i32 %conv13, %conv4
  %conv2 = trunc i32 %sub to i16
  store i16 %conv2, ptr %p, align 2
  ret void
}

define void @memop_signed_short_or(ptr nocapture %p, i16 signext %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_short_or:
; CHECK:  memh(r{{[0-9]+}}+#0) |= r{{[0-9]+}}
  %0 = load i16, ptr %p, align 2
  %or3 = or i16 %0, %x
  store i16 %or3, ptr %p, align 2
  ret void
}

define void @memop_signed_short_and(ptr nocapture %p, i16 signext %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_short_and:
; CHECK:  memh(r{{[0-9]+}}+#0) &= r{{[0-9]+}}
  %0 = load i16, ptr %p, align 2
  %and3 = and i16 %0, %x
  store i16 %and3, ptr %p, align 2
  ret void
}

define void @memop_signed_short_clrbit(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_signed_short_clrbit:
; CHECK:  memh(r{{[0-9]+}}+#0) = clrbit(#5)
  %0 = load i16, ptr %p, align 2
  %conv2 = zext i16 %0 to i32
  %and = and i32 %conv2, 65503
  %conv1 = trunc i32 %and to i16
  store i16 %conv1, ptr %p, align 2
  ret void
}

define void @memop_signed_short_setbit(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_signed_short_setbit:
; CHECK:  memh(r{{[0-9]+}}+#0) = setbit(#7)
  %0 = load i16, ptr %p, align 2
  %conv2 = zext i16 %0 to i32
  %or = or i32 %conv2, 128
  %conv1 = trunc i32 %or to i16
  store i16 %conv1, ptr %p, align 2
  ret void
}

define void @memop_signed_short_add5_index(ptr nocapture %p, i32 %i) nounwind {
entry:
; CHECK-LABEL: memop_signed_short_add5_index:
; CHECK:  memh(r{{[0-9]+}}+#0) += #5
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 %i
  %0 = load i16, ptr %add.ptr, align 2
  %conv2 = zext i16 %0 to i32
  %add = add nsw i32 %conv2, 5
  %conv1 = trunc i32 %add to i16
  store i16 %conv1, ptr %add.ptr, align 2
  ret void
}

define void @memop_signed_short_add_index(ptr nocapture %p, i32 %i, i16 signext %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_short_add_index:
; CHECK:  memh(r{{[0-9]+}}+#0) += r{{[0-9]+}}
  %conv4 = zext i16 %x to i32
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 %i
  %0 = load i16, ptr %add.ptr, align 2
  %conv13 = zext i16 %0 to i32
  %add = add nsw i32 %conv13, %conv4
  %conv2 = trunc i32 %add to i16
  store i16 %conv2, ptr %add.ptr, align 2
  ret void
}

define void @memop_signed_short_sub_index(ptr nocapture %p, i32 %i, i16 signext %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_short_sub_index:
; CHECK:  memh(r{{[0-9]+}}+#0) -= r{{[0-9]+}}
  %conv4 = zext i16 %x to i32
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 %i
  %0 = load i16, ptr %add.ptr, align 2
  %conv13 = zext i16 %0 to i32
  %sub = sub nsw i32 %conv13, %conv4
  %conv2 = trunc i32 %sub to i16
  store i16 %conv2, ptr %add.ptr, align 2
  ret void
}

define void @memop_signed_short_or_index(ptr nocapture %p, i32 %i, i16 signext %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_short_or_index:
; CHECK:  memh(r{{[0-9]+}}+#0) |= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 %i
  %0 = load i16, ptr %add.ptr, align 2
  %or3 = or i16 %0, %x
  store i16 %or3, ptr %add.ptr, align 2
  ret void
}

define void @memop_signed_short_and_index(ptr nocapture %p, i32 %i, i16 signext %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_short_and_index:
; CHECK:  memh(r{{[0-9]+}}+#0) &= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 %i
  %0 = load i16, ptr %add.ptr, align 2
  %and3 = and i16 %0, %x
  store i16 %and3, ptr %add.ptr, align 2
  ret void
}

define void @memop_signed_short_clrbit_index(ptr nocapture %p, i32 %i) nounwind {
entry:
; CHECK-LABEL: memop_signed_short_clrbit_index:
; CHECK:  memh(r{{[0-9]+}}+#0) = clrbit(#5)
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 %i
  %0 = load i16, ptr %add.ptr, align 2
  %conv2 = zext i16 %0 to i32
  %and = and i32 %conv2, 65503
  %conv1 = trunc i32 %and to i16
  store i16 %conv1, ptr %add.ptr, align 2
  ret void
}

define void @memop_signed_short_setbit_index(ptr nocapture %p, i32 %i) nounwind {
entry:
; CHECK-LABEL: memop_signed_short_setbit_index:
; CHECK:  memh(r{{[0-9]+}}+#0) = setbit(#7)
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 %i
  %0 = load i16, ptr %add.ptr, align 2
  %conv2 = zext i16 %0 to i32
  %or = or i32 %conv2, 128
  %conv1 = trunc i32 %or to i16
  store i16 %conv1, ptr %add.ptr, align 2
  ret void
}

define void @memop_signed_short_add5_index5(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_signed_short_add5_index5:
; CHECK:  memh(r{{[0-9]+}}+#10) += #5
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 5
  %0 = load i16, ptr %add.ptr, align 2
  %conv2 = zext i16 %0 to i32
  %add = add nsw i32 %conv2, 5
  %conv1 = trunc i32 %add to i16
  store i16 %conv1, ptr %add.ptr, align 2
  ret void
}

define void @memop_signed_short_add_index5(ptr nocapture %p, i16 signext %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_short_add_index5:
; CHECK:  memh(r{{[0-9]+}}+#10) += r{{[0-9]+}}
  %conv4 = zext i16 %x to i32
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 5
  %0 = load i16, ptr %add.ptr, align 2
  %conv13 = zext i16 %0 to i32
  %add = add nsw i32 %conv13, %conv4
  %conv2 = trunc i32 %add to i16
  store i16 %conv2, ptr %add.ptr, align 2
  ret void
}

define void @memop_signed_short_sub_index5(ptr nocapture %p, i16 signext %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_short_sub_index5:
; CHECK:  memh(r{{[0-9]+}}+#10) -= r{{[0-9]+}}
  %conv4 = zext i16 %x to i32
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 5
  %0 = load i16, ptr %add.ptr, align 2
  %conv13 = zext i16 %0 to i32
  %sub = sub nsw i32 %conv13, %conv4
  %conv2 = trunc i32 %sub to i16
  store i16 %conv2, ptr %add.ptr, align 2
  ret void
}

define void @memop_signed_short_or_index5(ptr nocapture %p, i16 signext %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_short_or_index5:
; CHECK:  memh(r{{[0-9]+}}+#10) |= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 5
  %0 = load i16, ptr %add.ptr, align 2
  %or3 = or i16 %0, %x
  store i16 %or3, ptr %add.ptr, align 2
  ret void
}

define void @memop_signed_short_and_index5(ptr nocapture %p, i16 signext %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_short_and_index5:
; CHECK:  memh(r{{[0-9]+}}+#10) &= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 5
  %0 = load i16, ptr %add.ptr, align 2
  %and3 = and i16 %0, %x
  store i16 %and3, ptr %add.ptr, align 2
  ret void
}

define void @memop_signed_short_clrbit_index5(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_signed_short_clrbit_index5:
; CHECK:  memh(r{{[0-9]+}}+#10) = clrbit(#5)
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 5
  %0 = load i16, ptr %add.ptr, align 2
  %conv2 = zext i16 %0 to i32
  %and = and i32 %conv2, 65503
  %conv1 = trunc i32 %and to i16
  store i16 %conv1, ptr %add.ptr, align 2
  ret void
}

define void @memop_signed_short_setbit_index5(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_signed_short_setbit_index5:
; CHECK:  memh(r{{[0-9]+}}+#10) = setbit(#7)
  %add.ptr = getelementptr inbounds i16, ptr %p, i32 5
  %0 = load i16, ptr %add.ptr, align 2
  %conv2 = zext i16 %0 to i32
  %or = or i32 %conv2, 128
  %conv1 = trunc i32 %or to i16
  store i16 %conv1, ptr %add.ptr, align 2
  ret void
}

define void @memop_signed_int_add5(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_signed_int_add5:
; CHECK:  memw(r{{[0-9]+}}+#0) += #5
  %0 = load i32, ptr %p, align 4
  %add = add i32 %0, 5
  store i32 %add, ptr %p, align 4
  ret void
}

define void @memop_signed_int_add(ptr nocapture %p, i32 %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_int_add:
; CHECK:  memw(r{{[0-9]+}}+#0) += r{{[0-9]+}}
  %0 = load i32, ptr %p, align 4
  %add = add i32 %0, %x
  store i32 %add, ptr %p, align 4
  ret void
}

define void @memop_signed_int_sub(ptr nocapture %p, i32 %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_int_sub:
; CHECK:  memw(r{{[0-9]+}}+#0) -= r{{[0-9]+}}
  %0 = load i32, ptr %p, align 4
  %sub = sub i32 %0, %x
  store i32 %sub, ptr %p, align 4
  ret void
}

define void @memop_signed_int_or(ptr nocapture %p, i32 %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_int_or:
; CHECK:  memw(r{{[0-9]+}}+#0) |= r{{[0-9]+}}
  %0 = load i32, ptr %p, align 4
  %or = or i32 %0, %x
  store i32 %or, ptr %p, align 4
  ret void
}

define void @memop_signed_int_and(ptr nocapture %p, i32 %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_int_and:
; CHECK:  memw(r{{[0-9]+}}+#0) &= r{{[0-9]+}}
  %0 = load i32, ptr %p, align 4
  %and = and i32 %0, %x
  store i32 %and, ptr %p, align 4
  ret void
}

define void @memop_signed_int_clrbit(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_signed_int_clrbit:
; CHECK:  memw(r{{[0-9]+}}+#0) = clrbit(#5)
  %0 = load i32, ptr %p, align 4
  %and = and i32 %0, -33
  store i32 %and, ptr %p, align 4
  ret void
}

define void @memop_signed_int_setbit(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_signed_int_setbit:
; CHECK:  memw(r{{[0-9]+}}+#0) = setbit(#7)
  %0 = load i32, ptr %p, align 4
  %or = or i32 %0, 128
  store i32 %or, ptr %p, align 4
  ret void
}

define void @memop_signed_int_add5_index(ptr nocapture %p, i32 %i) nounwind {
entry:
; CHECK-LABEL: memop_signed_int_add5_index:
; CHECK:  memw(r{{[0-9]+}}+#0) += #5
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 %i
  %0 = load i32, ptr %add.ptr, align 4
  %add = add i32 %0, 5
  store i32 %add, ptr %add.ptr, align 4
  ret void
}

define void @memop_signed_int_add_index(ptr nocapture %p, i32 %i, i32 %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_int_add_index:
; CHECK:  memw(r{{[0-9]+}}+#0) += r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 %i
  %0 = load i32, ptr %add.ptr, align 4
  %add = add i32 %0, %x
  store i32 %add, ptr %add.ptr, align 4
  ret void
}

define void @memop_signed_int_sub_index(ptr nocapture %p, i32 %i, i32 %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_int_sub_index:
; CHECK:  memw(r{{[0-9]+}}+#0) -= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 %i
  %0 = load i32, ptr %add.ptr, align 4
  %sub = sub i32 %0, %x
  store i32 %sub, ptr %add.ptr, align 4
  ret void
}

define void @memop_signed_int_or_index(ptr nocapture %p, i32 %i, i32 %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_int_or_index:
; CHECK:  memw(r{{[0-9]+}}+#0) |= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 %i
  %0 = load i32, ptr %add.ptr, align 4
  %or = or i32 %0, %x
  store i32 %or, ptr %add.ptr, align 4
  ret void
}

define void @memop_signed_int_and_index(ptr nocapture %p, i32 %i, i32 %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_int_and_index:
; CHECK:  memw(r{{[0-9]+}}+#0) &= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 %i
  %0 = load i32, ptr %add.ptr, align 4
  %and = and i32 %0, %x
  store i32 %and, ptr %add.ptr, align 4
  ret void
}

define void @memop_signed_int_clrbit_index(ptr nocapture %p, i32 %i) nounwind {
entry:
; CHECK-LABEL: memop_signed_int_clrbit_index:
; CHECK:  memw(r{{[0-9]+}}+#0) = clrbit(#5)
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 %i
  %0 = load i32, ptr %add.ptr, align 4
  %and = and i32 %0, -33
  store i32 %and, ptr %add.ptr, align 4
  ret void
}

define void @memop_signed_int_setbit_index(ptr nocapture %p, i32 %i) nounwind {
entry:
; CHECK-LABEL: memop_signed_int_setbit_index:
; CHECK:  memw(r{{[0-9]+}}+#0) = setbit(#7)
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 %i
  %0 = load i32, ptr %add.ptr, align 4
  %or = or i32 %0, 128
  store i32 %or, ptr %add.ptr, align 4
  ret void
}

define void @memop_signed_int_add5_index5(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_signed_int_add5_index5:
; CHECK:  memw(r{{[0-9]+}}+#20) += #5
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 5
  %0 = load i32, ptr %add.ptr, align 4
  %add = add i32 %0, 5
  store i32 %add, ptr %add.ptr, align 4
  ret void
}

define void @memop_signed_int_add_index5(ptr nocapture %p, i32 %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_int_add_index5:
; CHECK:  memw(r{{[0-9]+}}+#20) += r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 5
  %0 = load i32, ptr %add.ptr, align 4
  %add = add i32 %0, %x
  store i32 %add, ptr %add.ptr, align 4
  ret void
}

define void @memop_signed_int_sub_index5(ptr nocapture %p, i32 %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_int_sub_index5:
; CHECK:  memw(r{{[0-9]+}}+#20) -= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 5
  %0 = load i32, ptr %add.ptr, align 4
  %sub = sub i32 %0, %x
  store i32 %sub, ptr %add.ptr, align 4
  ret void
}

define void @memop_signed_int_or_index5(ptr nocapture %p, i32 %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_int_or_index5:
; CHECK:  memw(r{{[0-9]+}}+#20) |= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 5
  %0 = load i32, ptr %add.ptr, align 4
  %or = or i32 %0, %x
  store i32 %or, ptr %add.ptr, align 4
  ret void
}

define void @memop_signed_int_and_index5(ptr nocapture %p, i32 %x) nounwind {
entry:
; CHECK-LABEL: memop_signed_int_and_index5:
; CHECK:  memw(r{{[0-9]+}}+#20) &= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 5
  %0 = load i32, ptr %add.ptr, align 4
  %and = and i32 %0, %x
  store i32 %and, ptr %add.ptr, align 4
  ret void
}

define void @memop_signed_int_clrbit_index5(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_signed_int_clrbit_index5:
; CHECK:  memw(r{{[0-9]+}}+#20) = clrbit(#5)
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 5
  %0 = load i32, ptr %add.ptr, align 4
  %and = and i32 %0, -33
  store i32 %and, ptr %add.ptr, align 4
  ret void
}

define void @memop_signed_int_setbit_index5(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_signed_int_setbit_index5:
; CHECK:  memw(r{{[0-9]+}}+#20) = setbit(#7)
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 5
  %0 = load i32, ptr %add.ptr, align 4
  %or = or i32 %0, 128
  store i32 %or, ptr %add.ptr, align 4
  ret void
}

define void @memop_unsigned_int_add5(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_int_add5:
; CHECK:  memw(r{{[0-9]+}}+#0) += #5
  %0 = load i32, ptr %p, align 4
  %add = add nsw i32 %0, 5
  store i32 %add, ptr %p, align 4
  ret void
}

define void @memop_unsigned_int_add(ptr nocapture %p, i32 %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_int_add:
; CHECK:  memw(r{{[0-9]+}}+#0) += r{{[0-9]+}}
  %0 = load i32, ptr %p, align 4
  %add = add nsw i32 %0, %x
  store i32 %add, ptr %p, align 4
  ret void
}

define void @memop_unsigned_int_sub(ptr nocapture %p, i32 %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_int_sub:
; CHECK:  memw(r{{[0-9]+}}+#0) -= r{{[0-9]+}}
  %0 = load i32, ptr %p, align 4
  %sub = sub nsw i32 %0, %x
  store i32 %sub, ptr %p, align 4
  ret void
}

define void @memop_unsigned_int_or(ptr nocapture %p, i32 %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_int_or:
; CHECK:  memw(r{{[0-9]+}}+#0) |= r{{[0-9]+}}
  %0 = load i32, ptr %p, align 4
  %or = or i32 %0, %x
  store i32 %or, ptr %p, align 4
  ret void
}

define void @memop_unsigned_int_and(ptr nocapture %p, i32 %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_int_and:
; CHECK:  memw(r{{[0-9]+}}+#0) &= r{{[0-9]+}}
  %0 = load i32, ptr %p, align 4
  %and = and i32 %0, %x
  store i32 %and, ptr %p, align 4
  ret void
}

define void @memop_unsigned_int_clrbit(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_int_clrbit:
; CHECK:  memw(r{{[0-9]+}}+#0) = clrbit(#5)
  %0 = load i32, ptr %p, align 4
  %and = and i32 %0, -33
  store i32 %and, ptr %p, align 4
  ret void
}

define void @memop_unsigned_int_setbit(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_int_setbit:
; CHECK:  memw(r{{[0-9]+}}+#0) = setbit(#7)
  %0 = load i32, ptr %p, align 4
  %or = or i32 %0, 128
  store i32 %or, ptr %p, align 4
  ret void
}

define void @memop_unsigned_int_add5_index(ptr nocapture %p, i32 %i) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_int_add5_index:
; CHECK:  memw(r{{[0-9]+}}+#0) += #5
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 %i
  %0 = load i32, ptr %add.ptr, align 4
  %add = add nsw i32 %0, 5
  store i32 %add, ptr %add.ptr, align 4
  ret void
}

define void @memop_unsigned_int_add_index(ptr nocapture %p, i32 %i, i32 %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_int_add_index:
; CHECK:  memw(r{{[0-9]+}}+#0) += r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 %i
  %0 = load i32, ptr %add.ptr, align 4
  %add = add nsw i32 %0, %x
  store i32 %add, ptr %add.ptr, align 4
  ret void
}

define void @memop_unsigned_int_sub_index(ptr nocapture %p, i32 %i, i32 %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_int_sub_index:
; CHECK:  memw(r{{[0-9]+}}+#0) -= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 %i
  %0 = load i32, ptr %add.ptr, align 4
  %sub = sub nsw i32 %0, %x
  store i32 %sub, ptr %add.ptr, align 4
  ret void
}

define void @memop_unsigned_int_or_index(ptr nocapture %p, i32 %i, i32 %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_int_or_index:
; CHECK:  memw(r{{[0-9]+}}+#0) |= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 %i
  %0 = load i32, ptr %add.ptr, align 4
  %or = or i32 %0, %x
  store i32 %or, ptr %add.ptr, align 4
  ret void
}

define void @memop_unsigned_int_and_index(ptr nocapture %p, i32 %i, i32 %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_int_and_index:
; CHECK:  memw(r{{[0-9]+}}+#0) &= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 %i
  %0 = load i32, ptr %add.ptr, align 4
  %and = and i32 %0, %x
  store i32 %and, ptr %add.ptr, align 4
  ret void
}

define void @memop_unsigned_int_clrbit_index(ptr nocapture %p, i32 %i) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_int_clrbit_index:
; CHECK:  memw(r{{[0-9]+}}+#0) = clrbit(#5)
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 %i
  %0 = load i32, ptr %add.ptr, align 4
  %and = and i32 %0, -33
  store i32 %and, ptr %add.ptr, align 4
  ret void
}

define void @memop_unsigned_int_setbit_index(ptr nocapture %p, i32 %i) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_int_setbit_index:
; CHECK:  memw(r{{[0-9]+}}+#0) = setbit(#7)
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 %i
  %0 = load i32, ptr %add.ptr, align 4
  %or = or i32 %0, 128
  store i32 %or, ptr %add.ptr, align 4
  ret void
}

define void @memop_unsigned_int_add5_index5(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_int_add5_index5:
; CHECK:  memw(r{{[0-9]+}}+#20) += #5
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 5
  %0 = load i32, ptr %add.ptr, align 4
  %add = add nsw i32 %0, 5
  store i32 %add, ptr %add.ptr, align 4
  ret void
}

define void @memop_unsigned_int_add_index5(ptr nocapture %p, i32 %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_int_add_index5:
; CHECK:  memw(r{{[0-9]+}}+#20) += r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 5
  %0 = load i32, ptr %add.ptr, align 4
  %add = add nsw i32 %0, %x
  store i32 %add, ptr %add.ptr, align 4
  ret void
}

define void @memop_unsigned_int_sub_index5(ptr nocapture %p, i32 %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_int_sub_index5:
; CHECK:  memw(r{{[0-9]+}}+#20) -= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 5
  %0 = load i32, ptr %add.ptr, align 4
  %sub = sub nsw i32 %0, %x
  store i32 %sub, ptr %add.ptr, align 4
  ret void
}

define void @memop_unsigned_int_or_index5(ptr nocapture %p, i32 %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_int_or_index5:
; CHECK:  memw(r{{[0-9]+}}+#20) |= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 5
  %0 = load i32, ptr %add.ptr, align 4
  %or = or i32 %0, %x
  store i32 %or, ptr %add.ptr, align 4
  ret void
}

define void @memop_unsigned_int_and_index5(ptr nocapture %p, i32 %x) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_int_and_index5:
; CHECK:  memw(r{{[0-9]+}}+#20) &= r{{[0-9]+}}
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 5
  %0 = load i32, ptr %add.ptr, align 4
  %and = and i32 %0, %x
  store i32 %and, ptr %add.ptr, align 4
  ret void
}

define void @memop_unsigned_int_clrbit_index5(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_int_clrbit_index5:
; CHECK:  memw(r{{[0-9]+}}+#20) = clrbit(#5)
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 5
  %0 = load i32, ptr %add.ptr, align 4
  %and = and i32 %0, -33
  store i32 %and, ptr %add.ptr, align 4
  ret void
}

define void @memop_unsigned_int_setbit_index5(ptr nocapture %p) nounwind {
entry:
; CHECK-LABEL: memop_unsigned_int_setbit_index5:
; CHECK:  memw(r{{[0-9]+}}+#20) = setbit(#7)
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 5
  %0 = load i32, ptr %add.ptr, align 4
  %or = or i32 %0, 128
  store i32 %or, ptr %add.ptr, align 4
  ret void
}
