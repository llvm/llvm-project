import SwiftUI
import AppKit

struct TestView: View {
    @State var count: Int = 41

    var body: some View {
        count = 15 // This assignment does not become visible until after the graph update.
        print("break body")
        return Text("\(count)")
            .onAppear { [self]
                // count's value will be 15 at this point.
                print("break after")
                count = 23
                print("break final")
            }
    }
}

@main enum Entry {
    static func main() {
        let view = TestView()
        print("break before")

        // Cause the SwiftUI graph to call `body`, without showing a window.
        let hostingView = NSHostingView(rootView: view)
        hostingView.frame = .zero
        hostingView.layout()
    }
}
