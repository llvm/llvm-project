; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a %s -o /dev/null

; Regression test ensuring that empty subranges are not prematurely eliminated.

declare <16 x i63> @llvm.masked.load.v16i63.p0(ptr, <16 x i1>, <16 x i63>)

define void @f(ptr %p, <16 x i1> %m, <16 x i63> %pt, <16 x i1> %sc,
               <16 x double> %da, <16 x double> %db,
               <16 x i63> %v, <16 x i63> %w, <16 x i63> %a,
               ptr %sink, i32 %disc, i1 %c) #0 {
entry:
  switch i32 %disc, label %bb0 [
    i32 2, label %bb1
    i32 3, label %bb2
  ]
bb0:
  br i1 %c, label %bb0a, label %bb0b
bb0a:
  store <16 x i63> %a, ptr %sink
  ret void
bb0b:
  %ld = call <16 x i63> @llvm.masked.load.v16i63.p0(ptr %p, <16 x i1> %m, <16 x i63> %pt)
  store <16 x i63> %ld, ptr %sink
  ret void
bb1:
  br i1 %c, label %bb1a, label %bb1b
bb1a:
  %sel = select <16 x i1> %sc, <16 x double> %da, <16 x double> %db
  store <16 x double> %sel, ptr %sink
  ret void
bb1b:
  store <16 x i63> %v, ptr %sink
  ret void
bb2:
  %x = xor <16 x i63> %a, %w
  store <16 x i63> %x, ptr %sink
  ret void
}

attributes #0 = { nounwind }
