func f() {
    let array: [Int] = [10, 20, 30]
    array.withUnsafeBytes { buffer in
        print("break here") // break here
    }
}

f()
