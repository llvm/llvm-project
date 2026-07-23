import Lib

func inspect(_ h: Holder, _ p: Plain) {
    print(h.tag) // Set breakpoint here
}

inspect(Holder(), Plain())
