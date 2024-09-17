import Foundation

func main() {
  var notification = Notification(name: Notification.Name(rawValue: "MyNotification"), object: nil, userInfo: [:])
  print("break here!")
}

var g_notification = Notification(name: Notification.Name(rawValue: "MyNotification"), object: nil, userInfo: [:])

main()
