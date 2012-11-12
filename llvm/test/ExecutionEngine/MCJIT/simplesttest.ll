; RUN: %lli -mtriple=%mcjit_triple -use-mcjit %s > /dev/null

define i32 @main() {
	ret i32 0
}

