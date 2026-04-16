import SwiftUI
import AppKit

struct TestView: View {
    @State var count: Int = 41

    var body: some View {
        print("break body")
        return Text("\(count)")
            .onAppear {
                print("break appear")
                count = 23
                print("break change")
            }
            .onChange(of: count) {
                print("break change", self)
            }
    }
}

@main enum Entry {
    static func main() {
        // Cause the SwiftUI graph to call `body`, without showing a window.
        let view = TestView()
        let hostingView = NSHostingView(rootView: view)
        hostingView.frame = .zero
        hostingView.layout()
    }
}
