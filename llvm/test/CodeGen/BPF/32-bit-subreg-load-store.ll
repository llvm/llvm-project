; RUN: llc -O2 -mtriple=bpfel -mattr=+alu32 < %s | FileCheck %s
;
; unsigned char loadu8(unsigned char *p)
; {
;   return *p;
; }
;
; unsigned short loadu16(unsigned short *p)
; {
;   return *p;
; }
;
; unsigned loadu32(unsigned *p)
; {
;   return *p;
; }
;
; unsigned long long loadu64(unsigned long long *p)
; {
;   return *p;
; }
;
; void storeu8(unsigned char *p, unsigned long long v)
; {
;   *p = (unsigned char)v;
; }
;
; void storeu16(unsigned short *p, unsigned long long v)
; {
;   *p = (unsigned short)v;
; }
;
; void storeu32(unsigned *p, unsigned long long v)
; {
;   *p = (unsigned)v;
; }
;
; void storeu64(unsigned long long *p, unsigned long long v)
; {
;   *p = v;
; }
; Function Attrs: norecurse nounwind readonly
define dso_local zeroext i8 @loadu8(ptr nocapture readonly %p) local_unnamed_addr #0 {
entry:
  %0 = load i8, ptr %p, align 1
; CHECK: w{{[0-9]+}} = *(u8 *)(r{{[0-9]+}} + 0)
  ret i8 %0
}

; Function Attrs: norecurse nounwind readonly
define dso_local zeroext i16 @loadu16(ptr nocapture readonly %p) local_unnamed_addr #0 {
entry:
  %0 = load i16, ptr %p, align 2
; CHECK: w{{[0-9]+}} = *(u16 *)(r{{[0-9]+}} + 0)
  ret i16 %0
}

; Function Attrs: norecurse nounwind readonly
define dso_local i32 @loadu32(ptr nocapture readonly %p) local_unnamed_addr #0 {
entry:
  %0 = load i32, ptr %p, align 4
; CHECK: w{{[0-9]+}} = *(u32 *)(r{{[0-9]+}} + 0)
  ret i32 %0
}

; Function Attrs: norecurse nounwind readonly
define dso_local i64 @loadu64(ptr nocapture readonly %p) local_unnamed_addr #0 {
entry:
  %0 = load i64, ptr %p, align 8
; CHECK: r{{[0-9]+}} = *(u64 *)(r{{[0-9]+}} + 0)
  ret i64 %0
}

; Function Attrs: norecurse nounwind
define dso_local void @storeu8(ptr nocapture %p, i64 %v) local_unnamed_addr #1 {
entry:
  %conv = trunc i64 %v to i8
  store i8 %conv, ptr %p, align 1
; CHECK: *(u8 *)(r{{[0-9]+}} + 0) = w{{[0-9]+}}
  ret void
}

; Function Attrs: norecurse nounwind
define dso_local void @storeu16(ptr nocapture %p, i64 %v) local_unnamed_addr #1 {
entry:
  %conv = trunc i64 %v to i16
  store i16 %conv, ptr %p, align 2
; CHECK: *(u16 *)(r{{[0-9]+}} + 0) = w{{[0-9]+}}
  ret void
}

; Function Attrs: norecurse nounwind
define dso_local void @storeu32(ptr nocapture %p, i64 %v) local_unnamed_addr #1 {
entry:
  %conv = trunc i64 %v to i32
  store i32 %conv, ptr %p, align 4
; CHECK: *(u32 *)(r{{[0-9]+}} + 0) = w{{[0-9]+}}
  ret void
}

; Function Attrs: norecurse nounwind
define dso_local void @storeu64(ptr nocapture %p, i64 %v) local_unnamed_addr #1 {
entry:
  store i64 %v, ptr %p, align 8
; CHECK: *(u64 *)(r{{[0-9]+}} + 0) = r{{[0-9]+}}
  ret void
}
