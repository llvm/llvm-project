// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 0
// CHECK-NEXT: 1
// CHECK-NEXT: 2
// CHECK-NEXT: 3
// CHECK-NEXT: 4
// CHECK-NEXT: 5
// CHECK-NEXT: 6
// CHECK-NEXT: 7
// CHECK-NEXT: 8
// CHECK-NEXT: 9
// CHECK-NEXT: done
// CHECK-NEXT: !

package main

func f1() {
	goto labeled
labeled:
	goto done
	return
done:
	println("!")
}

func main() {
	i := 0
start:
	if i < 10 {
		println(i)
		i++
		goto start
	} else {
		goto end
	}
	return
end:
	println("done")
	f1()
	return
}
