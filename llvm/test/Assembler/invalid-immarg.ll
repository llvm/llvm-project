; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: Attribute 'immarg' is incompatible with other attributes
declare void @llvm.immarg.byval(i32* byval(i32) immarg)

; CHECK: Attribute 'immarg' is incompatible with other attributes
declare void @llvm.immarg.inalloca(i32* inalloca immarg)

; CHECK: Attribute 'immarg' is incompatible with other attributes
declare void @llvm.immarg.inreg(i32 inreg immarg)

; CHECK: Attribute 'immarg' is incompatible with other attributes
declare void @llvm.immarg.nest(i32* nest immarg)

; CHECK: Attribute 'immarg' is incompatible with other attributes
declare void @llvm.immarg.sret(i32* sret(i32) immarg)

; CHECK: Attribute 'immarg' is incompatible with other attributes
declare void @llvm.immarg.zeroext(i32 zeroext immarg)

; CHECK: Attribute 'immarg' is incompatible with other attributes
declare void @llvm.immarg.signext(i32 signext immarg)

; CHECK: Attribute 'immarg' is incompatible with other attributes
declare void @llvm.immarg.returned(i32 returned immarg)

; CHECK: Attribute 'immarg' is incompatible with other attributes
declare void @llvm.immarg.noalias(i32* noalias immarg)

; CHECK: Attribute 'immarg' is incompatible with other attributes
declare void @llvm.immarg.readnone(i32* readnone immarg)

; CHECK: Attribute 'immarg' is incompatible with other attributes
declare void @llvm.immarg.readonly(i32* readonly immarg)
