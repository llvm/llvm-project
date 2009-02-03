; RUN: llvm-as < %s | opt -sretpromotion -disable-output
	%struct.ObjPoint = type { double, double, double, double, double, double }

define void @RotatePoint(%struct.ObjPoint* sret  %agg.result, %struct.ObjPoint* byval  %a, double %rx, double %ry, double %rz) nounwind  {
entry:
	unreachable
}
