import Foundation

func main() {
  var name = Notification.Name(rawValue: "MyNotification")
  var notification = Notification(name: name, object: nil, userInfo: [:])
  print("break here!")
}

var g_notification = Notification(name: Notification.Name(rawValue: "MyNotification"), object: nil, userInfo: [:])

main()
