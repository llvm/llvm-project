import OpaqueLib

func use() {
    let manager = Manager()
    let variable = manager.producer
    print("break here") // break here
    _ = variable
}

use()
