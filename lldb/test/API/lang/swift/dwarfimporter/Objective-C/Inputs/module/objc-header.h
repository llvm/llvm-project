/* -*- ObjC -*- */
@import Foundation;

@protocol WithName
- (NSString *)name;
@end

@protocol ObjCProtocol <WithName>
@end

@interface ObjCClass : NSObject
@property (readonly) int number;
- (instancetype)init;
@end

// FIXME: id<ObjCProtocol> doesn't anchor the type since Clang doesn't
// yet describe protocol conformances in DWARF.
id<ObjCProtocol> getProto();
