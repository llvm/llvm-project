@_marker protocol MyMarker {}
@_marker protocol MyMarker2 {}

protocol MyProto {}

struct S: MyMarker {
    var x = 42
}

struct T: MyMarker, MyProto {
    var a = 10
}

struct U: MyMarker, MyMarker2 {
    var b = 20
}

struct V: MyMarker, MyMarker2, MyProto {
    var d = 30
}

func marker_only() {
    let v: any MyMarker = S()
    print(v) // break marker only
}

func marker_composition() {
    let v: any MyMarker & MyProto = T()
    print(v) // break composition
}

func two_markers() {
    let v: any MyMarker & MyMarker2 = U()
    print(v) // break two markers
}

func any_and_marker() {
    let v: any Any & MyMarker = S()
    print(v) // break any and marker
}

func any_marker_and_non_marker() {
    let v: any Any & MyMarker & MyProto = V()
    print(v) // break any marker and non marker
}

marker_only()
marker_composition()
two_markers()
any_and_marker()
any_marker_and_non_marker()
