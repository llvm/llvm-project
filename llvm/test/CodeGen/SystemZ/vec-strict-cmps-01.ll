; Test signaling f32 and v4f32 comparisons on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

; Test oeq.
define <4 x i32> @f1(<4 x i32> %dummy, <4 x float> %val1, <4 x float> %val2) #0 {
; CHECK-LABEL: f1:
; CHECK: vfkesb %v24, %v26, %v28
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test one.
define <4 x i32> @f2(<4 x i32> %dummy, <4 x float> %val1, <4 x float> %val2) #0 {
; CHECK-LABEL: f2:
; CHECK-DAG: vfkhsb [[REG1:%v[0-9]+]], %v28, %v26
; CHECK-DAG: vfkhsb [[REG2:%v[0-9]+]], %v26, %v28
; CHECK: vo %v24, [[REG1]], [[REG2]]
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"one",
                                               metadata !"fpexcept.strict") #0
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test ogt.
define <4 x i32> @f3(<4 x i32> %dummy, <4 x float> %val1, <4 x float> %val2) #0 {
; CHECK-LABEL: f3:
; CHECK: vfkhsb %v24, %v26, %v28
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"ogt",
                                               metadata !"fpexcept.strict") #0
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test oge.
define <4 x i32> @f4(<4 x i32> %dummy, <4 x float> %val1, <4 x float> %val2) #0 {
; CHECK-LABEL: f4:
; CHECK: vfkhesb %v24, %v26, %v28
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"oge",
                                               metadata !"fpexcept.strict") #0
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test ole.
define <4 x i32> @f5(<4 x i32> %dummy, <4 x float> %val1, <4 x float> %val2) #0 {
; CHECK-LABEL: f5:
; CHECK: vfkhesb %v24, %v28, %v26
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"ole",
                                               metadata !"fpexcept.strict") #0
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test olt.
define <4 x i32> @f6(<4 x i32> %dummy, <4 x float> %val1, <4 x float> %val2) #0 {
; CHECK-LABEL: f6:
; CHECK: vfkhsb %v24, %v28, %v26
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test ueq.
define <4 x i32> @f7(<4 x i32> %dummy, <4 x float> %val1, <4 x float> %val2) #0 {
; CHECK-LABEL: f7:
; CHECK-DAG: vfkhsb [[REG1:%v[0-9]+]], %v28, %v26
; CHECK-DAG: vfkhsb [[REG2:%v[0-9]+]], %v26, %v28
; CHECK: vno %v24, [[REG1]], [[REG2]]
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"ueq",
                                               metadata !"fpexcept.strict") #0
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test une.
define <4 x i32> @f8(<4 x i32> %dummy, <4 x float> %val1, <4 x float> %val2) #0 {
; CHECK-LABEL: f8:
; CHECK: vfkesb [[REG:%v[0-9]+]], %v26, %v28
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"une",
                                               metadata !"fpexcept.strict") #0
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test ugt.
define <4 x i32> @f9(<4 x i32> %dummy, <4 x float> %val1, <4 x float> %val2) #0 {
; CHECK-LABEL: f9:
; CHECK: vfkhesb [[REG:%v[0-9]+]], %v28, %v26
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"ugt",
                                               metadata !"fpexcept.strict") #0
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test uge.
define <4 x i32> @f10(<4 x i32> %dummy, <4 x float> %val1,
                      <4 x float> %val2) #0 {
; CHECK-LABEL: f10:
; CHECK: vfkhsb [[REG:%v[0-9]+]], %v28, %v26
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"uge",
                                               metadata !"fpexcept.strict") #0
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test ule.
define <4 x i32> @f11(<4 x i32> %dummy, <4 x float> %val1,
                      <4 x float> %val2) #0 {
; CHECK-LABEL: f11:
; CHECK: vfkhsb [[REG:%v[0-9]+]], %v26, %v28
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"ule",
                                               metadata !"fpexcept.strict") #0
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test ult.
define <4 x i32> @f12(<4 x i32> %dummy, <4 x float> %val1,
                      <4 x float> %val2) #0 {
; CHECK-LABEL: f12:
; CHECK: vfkhesb [[REG:%v[0-9]+]], %v26, %v28
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"ult",
                                               metadata !"fpexcept.strict") #0
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test ord.
define <4 x i32> @f13(<4 x i32> %dummy, <4 x float> %val1,
                      <4 x float> %val2) #0 {
; CHECK-LABEL: f13:
; CHECK-DAG: vfkhsb [[REG1:%v[0-9]+]], %v28, %v26
; CHECK-DAG: vfkhesb [[REG2:%v[0-9]+]], %v26, %v28
; CHECK: vo %v24, [[REG1]], [[REG2]]
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"ord",
                                               metadata !"fpexcept.strict") #0
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test uno.
define <4 x i32> @f14(<4 x i32> %dummy, <4 x float> %val1,
                      <4 x float> %val2) #0 {
; CHECK-LABEL: f14:
; CHECK-DAG: vfkhsb [[REG1:%v[0-9]+]], %v28, %v26
; CHECK-DAG: vfkhesb [[REG2:%v[0-9]+]], %v26, %v28
; CHECK: vno %v24, [[REG1]], [[REG2]]
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"uno",
                                               metadata !"fpexcept.strict") #0
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test oeq selects.
define <4 x float> @f15(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) #0 {
; CHECK-LABEL: f15:
; CHECK: vfkesb [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test one selects.
define <4 x float> @f16(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) #0 {
; CHECK-LABEL: f16:
; CHECK-DAG: vfkhsb [[REG1:%v[0-9]+]], %v26, %v24
; CHECK-DAG: vfkhsb [[REG2:%v[0-9]+]], %v24, %v26
; CHECK: vo [[REG:%v[0-9]+]], [[REG1]], [[REG2]]
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"one",
                                               metadata !"fpexcept.strict") #0
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test ogt selects.
define <4 x float> @f17(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) #0 {
; CHECK-LABEL: f17:
; CHECK: vfkhsb [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"ogt",
                                               metadata !"fpexcept.strict") #0
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test oge selects.
define <4 x float> @f18(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) #0 {
; CHECK-LABEL: f18:
; CHECK: vfkhesb [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"oge",
                                               metadata !"fpexcept.strict") #0
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test ole selects.
define <4 x float> @f19(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) #0 {
; CHECK-LABEL: f19:
; CHECK: vfkhesb [[REG:%v[0-9]+]], %v26, %v24
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"ole",
                                               metadata !"fpexcept.strict") #0
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test olt selects.
define <4 x float> @f20(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) #0 {
; CHECK-LABEL: f20:
; CHECK: vfkhsb [[REG:%v[0-9]+]], %v26, %v24
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test ueq selects.
define <4 x float> @f21(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) #0 {
; CHECK-LABEL: f21:
; CHECK-DAG: vfkhsb [[REG1:%v[0-9]+]], %v26, %v24
; CHECK-DAG: vfkhsb [[REG2:%v[0-9]+]], %v24, %v26
; CHECK: vo [[REG:%v[0-9]+]], [[REG1]], [[REG2]]
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"ueq",
                                               metadata !"fpexcept.strict") #0
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test une selects.
define <4 x float> @f22(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) #0 {
; CHECK-LABEL: f22:
; CHECK: vfkesb [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"une",
                                               metadata !"fpexcept.strict") #0
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test ugt selects.
define <4 x float> @f23(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) #0 {
; CHECK-LABEL: f23:
; CHECK: vfkhesb [[REG:%v[0-9]+]], %v26, %v24
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"ugt",
                                               metadata !"fpexcept.strict") #0
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test uge selects.
define <4 x float> @f24(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) #0 {
; CHECK-LABEL: f24:
; CHECK: vfkhsb [[REG:%v[0-9]+]], %v26, %v24
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"uge",
                                               metadata !"fpexcept.strict") #0
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test ule selects.
define <4 x float> @f25(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) #0 {
; CHECK-LABEL: f25:
; CHECK: vfkhsb [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"ule",
                                               metadata !"fpexcept.strict") #0
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test ult selects.
define <4 x float> @f26(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) #0 {
; CHECK-LABEL: f26:
; CHECK: vfkhesb [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"ult",
                                               metadata !"fpexcept.strict") #0
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test ord selects.
define <4 x float> @f27(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) #0 {
; CHECK-LABEL: f27:
; CHECK-DAG: vfkhsb [[REG1:%v[0-9]+]], %v26, %v24
; CHECK-DAG: vfkhesb [[REG2:%v[0-9]+]], %v24, %v26
; CHECK: vo [[REG:%v[0-9]+]], [[REG1]], [[REG2]]
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"ord",
                                               metadata !"fpexcept.strict") #0
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test uno selects.
define <4 x float> @f28(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) #0 {
; CHECK-LABEL: f28:
; CHECK-DAG: vfkhsb [[REG1:%v[0-9]+]], %v26, %v24
; CHECK-DAG: vfkhesb [[REG2:%v[0-9]+]], %v24, %v26
; CHECK: vo [[REG:%v[0-9]+]], [[REG1]], [[REG2]]
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"uno",
                                               metadata !"fpexcept.strict") #0
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test an f32 comparison that uses vector registers.
define i64 @f29(i64 %a, i64 %b, float %f1, <4 x float> %vec) #0 {
; CHECK-LABEL: f29:
; CHECK: wfksb %f0, %v24
; CHECK-NEXT: locgrne %r2, %r3
; CHECK: br %r14
  %f2 = extractelement <4 x float> %vec, i32 0
  %cond = call i1 @llvm.experimental.constrained.fcmps.f32(
                                               float %f1, float %f2,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

attributes #0 = { strictfp }

declare <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float>, <4 x float>, metadata, metadata)
declare i1 @llvm.experimental.constrained.fcmps.f32(float, float, metadata, metadata)

