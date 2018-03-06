; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s | FileCheck %s
; Check that we generate store instructions with global + offset

%struct.struc = type { i8, i8, i16, i32 }

@foo = common global %struct.struc zeroinitializer, align 4

define void @storeByte(i32 %val1, i32 %val2, i8 zeroext %ival) nounwind {
; CHECK: memb(##foo+1) = r{{[0-9]+}}
entry:
  %cmp = icmp sgt i32 %val1, %val2
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i8 %ival, i8* getelementptr inbounds (%struct.struc, %struct.struc* @foo, i32 0, i32 1), align 1
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @storeHW(i32 %val1, i32 %val2, i16 signext %ival) nounwind {
; CHECK: memh(##foo+2) = r{{[0-9]+}}
entry:
  %cmp = icmp sgt i32 %val1, %val2
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i16 %ival, i16* getelementptr inbounds (%struct.struc, %struct.struc* @foo, i32 0, i32 2), align 2
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

