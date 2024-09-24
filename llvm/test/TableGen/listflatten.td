// RUN: llvm-tblgen %s | FileCheck %s

class Flatten<list<int> A, list<int> B> {
    list<int> Flat1 = !listflatten([A, B, [6], [7, 8]]);

    list<list<int>> X = [A, B];
    list<int> Flat2 = !listflatten(!listconcat(X, [[7]]));

    // Generate a nested list of integers.
    list<int> Y0 = [1, 2, 3, 4];
    list<list<int>> Y1 = !foreach(elem, Y0, [elem]);
    list<list<list<int>>> Y2 = !foreach(elem, Y1, [elem]);
    list<list<list<list<int>>>> Y3 = !foreach(elem, Y2, [elem]);

    // Flatten it completely.
    list<int> Flat3=!listflatten(!listflatten(!listflatten(Y3)));

    // Flatten it partially.
    list<list<list<int>>> Flat4 = !listflatten(Y3);
    list<list<int>> Flat5 = !listflatten(!listflatten(Y3));

    // Test NOP flattening.
    list<string> Flat6 = !listflatten(["a", "b"]);
}

// CHECK: list<int> Flat1 = [1, 2, 3, 4, 5, 6, 7, 8];
// CHECK: list<int> Flat2 = [1, 2, 3, 4, 5, 7];
// CHECK: list<int> Flat3 = [1, 2, 3, 4];
// CHECK{LITERAL}: list<list<list<int>>> Flat4 = [[[1]], [[2]], [[3]], [[4]]];
// CHECK: list<string> Flat6 = ["a", "b"];
def F : Flatten<[1,2], [3,4,5]>;

