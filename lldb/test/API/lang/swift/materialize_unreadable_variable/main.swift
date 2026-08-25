import Lib

// `unreadable` is never mentioned by the expressions the test evaluates. It is
// in scope, which is enough for the Swift expression parser to inject it into
// the expression wrapper and for the materializer to have to read it.
func inspect(_ unreadable: Unreadable, _ p: Plain) {
    print(p.value) // break here
}

inspect(Unreadable(tag: 42), Plain())
