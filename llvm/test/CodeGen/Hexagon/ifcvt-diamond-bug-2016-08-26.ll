; RUN: llc -march=hexagon -o - %s | FileCheck %s
target triple = "hexagon"

%struct.0 = type { i16, i16 }

@t = external local_unnamed_addr global %struct.0, align 2

define void @foo(i32 %p) local_unnamed_addr #0 {
entry:
  %conv90 = trunc i32 %p to i16
  %call105 = call signext i16 @bar(i16 signext 16384, i16 signext undef) #0
  %call175 = call signext i16 @bar(i16 signext %conv90, i16 signext 4) #0
  %call197 = call signext i16 @bar(i16 signext %conv90, i16 signext 4) #0
  %cmp199 = icmp eq i16 %call197, 0
  br i1 %cmp199, label %if.then200, label %if.else201

; CHECK-DAG: [[R4:r[0-9]+]] = #4
; CHECK: p0 = cmp.eq(r0, #0)
; CHECK: if (!p0.new) [[R3:r[0-9]+]] = #3
; CHECK-DAG: if (!p0) memh(##t) = [[R3]]
; CHECK-DAG: if (p0) memh(##t) = [[R4]]
if.then200:                                       ; preds = %entry
  store i16 4, i16* getelementptr inbounds (%struct.0, %struct.0* @t, i32 0, i32 0), align 2
  store i16 0, i16* getelementptr inbounds (%struct.0, %struct.0* @t, i32 0, i32 1), align 2
  br label %if.end202

if.else201:                                       ; preds = %entry
  store i16 3, i16* getelementptr inbounds (%struct.0, %struct.0* @t, i32 0, i32 0), align 2
  br label %if.end202

if.end202:                                        ; preds = %if.else201, %if.then200
  ret void
}

declare signext i16 @bar(i16 signext, i16 signext) local_unnamed_addr #0

attributes #0 = { optsize "target-cpu"="hexagonv55" }
