class Class {}

func use<T>(_ t: T) {}

func main() {
	let array = ContiguousArray<Class>([Class()])
	use(array)// Set breakpoint here
}

main()

