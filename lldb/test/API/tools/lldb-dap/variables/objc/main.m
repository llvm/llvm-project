#import <Foundation/Foundation.h>

@interface Greeter : NSObject

@property(nonatomic, strong) NSString *name;

- (void)greet:(NSString *)other;

@end

@implementation Greeter

- (instancetype)initWithName:(NSString *)name {
  if ((self = [super init])) {
    _name = [name copy];
  }
  return self;
}

- (void)greet:(NSString *)other {
  NSLog(@"Hello %@, from %@", other, _name);
}

- (NSString *)description {
  return
      [NSString stringWithFormat:@"<Greeter %p name=%@>", (void *)self, _name];
}

- (NSString *)debugDescription {
  return [NSString stringWithFormat:@"<Greeter %p name=%@ debugDescription>",
                                    (void *)self, _name];
}

@end

int main(int argc, char *argv[]) {
  Greeter *greeter = [[Greeter alloc] initWithName:@"Bob"];
  if (argc > 1) {
    [greeter greet:@(argv[1])];
  } else {
    [greeter greet:@"World"];
  }

  return 0; // breakpoint
}
