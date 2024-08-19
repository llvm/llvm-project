import Foundation

struct Payload: Decodable {
    var number: Int
}

func main() {
    do {
        let data = Data(#"{"numero":23}"#.utf8)
        _ = try JSONDecoder().decode(Payload.self, from: data)
    } catch {
        print("break here")
    }
}

main()
