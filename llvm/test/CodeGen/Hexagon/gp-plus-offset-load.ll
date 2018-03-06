; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s | FileCheck %s
; Check that we generate load instructions with global + offset

%struct.struc = type { i8, i8, i16, i32 }

@foo = common global %struct.struc zeroinitializer, align 4

define void @loadWord(i32 %val1, i32 %val2, i32* nocapture %ival) nounwind {
; CHECK: r{{[0-9]+}} = memw(##foo+4)
entry:
  %cmp = icmp sgt i32 %val1, %val2
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %0 = load i32, i32* getelementptr inbounds (%struct.struc, %struct.struc* @foo, i32 0, i32 3), align 4
  store i32 %0, i32* %ival, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @loadByte(i32 %val1, i32 %val2, i8* nocapture %ival) nounwind {
; CHECK: r{{[0-9]+}} = memub(##foo+1)
entry:
  %cmp = icmp sgt i32 %val1, %val2
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %0 = load i8, i8* getelementptr inbounds (%struct.struc, %struct.struc* @foo, i32 0, i32 1), align 1
  store i8 %0, i8* %ival, align 1
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @loadHWord(i32 %val1, i32 %val2, i16* %ival) nounwind {
; CHECK: r{{[0-9]+}} = memuh(##foo+2)
entry:
  %cmp = icmp sgt i32 %val1, %val2
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %0 = load i16, i16* getelementptr inbounds (%struct.struc, %struct.struc* @foo, i32 0, i32 2), align 2
  store i16 %0, i16* %ival, align 2
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}
