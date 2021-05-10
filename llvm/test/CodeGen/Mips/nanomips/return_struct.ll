; RUN: llc -mtriple=nanomips -verify-machineinstrs --stop-after=finalize-isel < %s | FileCheck %s

%struct.Test = type { i32, i32, i32, i32 }

define void @test_sret(%struct.Test* sret(%struct.Test) %agg.result) {
  %x = getelementptr inbounds %struct.Test, %struct.Test* %agg.result, i32 0, i32 0
  store i32 1, i32* %x
  %y = getelementptr inbounds %struct.Test, %struct.Test* %agg.result, i32 0, i32 1
  store i32 2, i32* %y
  %z = getelementptr inbounds %struct.Test, %struct.Test* %agg.result, i32 0, i32 2
  store i32 3, i32* %z
  %w = getelementptr inbounds %struct.Test, %struct.Test* %agg.result, i32 0, i32 3
  store i32 4, i32* %w
; Make sure that nanoMIPS register is used for returning struct by value.
; CHECK:RetRA implicit $a0_nm
  ret void
}
