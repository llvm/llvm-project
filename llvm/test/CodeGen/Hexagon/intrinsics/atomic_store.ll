; RUN: sed -e "s/ORDER/unordered/" %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/monotonic/" %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/release/" %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/seq_cst/" %s | llc -march=hexagon | FileCheck %s

%struct.Obj = type { [100 x i32] }

@i8Src   = global i8 0,  align 1
@i8Dest  = global i8 0,  align 1
@i16Src  = global i16 0, align 2
@i16Dest = global i16 0, align 2
@i32Src  = global i32 0, align 4
@i32Dest = global i32 0, align 4
@i64Src  = global i64 0, align 8
@i64Dest = global i64 0, align 8
@ptrSrc  = global ptr null, align 4
@ptrDest = global ptr null, align 4

define void @store_i8() #0 {
entry:
  %i8Tmp = load i8, ptr @i8Src, align 1
  store atomic i8 %i8Tmp, ptr @i8Dest ORDER, align 1
  ret void
}
; CHECK-LABEL: store_i8:
; CHECK: [[TMP_REG:r[0-9]+]] = memub(gp+#i8Src)
; CHECK: memb(gp+#i8Dest) = [[TMP_REG]]

define void @store_i16() #0 {
entry:
  %i16Tmp = load i16, ptr @i16Src, align 2
  store atomic i16 %i16Tmp, ptr @i16Dest ORDER, align 2
  ret void
}
; CHECK-LABEL: store_i16:
; CHECK: [[TMP_REG:r[0-9]+]] = memuh(gp+#i16Src)
; CHECK: memh(gp+#i16Dest) = [[TMP_REG]]

define void @store_i32() #0 {
entry:
  %i32Tmp = load i32, ptr @i32Src, align 4
  store atomic i32 %i32Tmp, ptr @i32Dest ORDER, align 4
  ret void
}
; CHECK-LABEL: store_i32:
; CHECK: [[TMP_REG:r[0-9]+]] = memw(gp+#i32Src)
; CHECK: memw(gp+#i32Dest) = [[TMP_REG]]

define void @store_i64() #0 {
entry:
  %i64Tmp = load i64, ptr @i64Src, align 8
  store atomic i64 %i64Tmp, ptr @i64Dest ORDER, align 8
  ret void
}
; CHECK-LABEL: store_i64:
; CHECK: [[TMP_REG:r[0-9]+:[0-9]+]] = memd(gp+#i64Src)
; CHECK: memd(gp+#i64Dest) = [[TMP_REG]]

define void @store_ptr() #0 {
entry:
  %ptrTmp = load i32, ptr @ptrSrc, align 4
  store atomic i32 %ptrTmp, ptr @ptrDest ORDER, align 4
  ret void
}
; CHECK-LABEL: store_ptr:
; CHECK: [[TMP_REG:r[0-9]+]] = memw(gp+#ptrSrc)
; CHECK: memw(gp+#ptrDest) = [[TMP_REG]]

