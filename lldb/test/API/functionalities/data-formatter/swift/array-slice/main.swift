func main() {
    let a = [1, 2, 3]
    let someSlice = a[1...]
    let arraySlice: ArraySlice<Int> = a[1...]
    let arraySubSequence: Array<Int>.SubSequence = a[1...]
    // break here
}

main()
