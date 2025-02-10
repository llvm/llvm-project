import WithDebInfo
import WithoutDebInfo

func withDebugInfo() {
    var s = WithDebInfo.S()
    print("break here with debug info")
}

func withoutDebugInfo() {
    var s = WithoutDebInfo.S()
    print("break here without debug info")
}

withDebugInfo()
withoutDebugInfo()
