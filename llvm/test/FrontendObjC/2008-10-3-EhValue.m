// RUN: %llvmgcc -w -x objective-c -c %s -o /dev/null

@interface Object {
@public
     Class isa;
}
+initialize;
+alloc;
+new;
+free;
-free;
+(Class)class;
-(Class)class;
-init;
-superclass;
-(const char *)name;
@end

@interface Frob: Object
@end

@implementation Frob: Object
@end

static Frob* _connection = ((void *)0);

extern void abort(void);

void test (Object* sendPort)
{
 int cleanupPorts = 1;
 Frob* receivePort = ((void *)0);

 @try {
  receivePort = (Frob *) -1;
  _connection = (Frob *) -1;
  receivePort = ((void *)0);
  sendPort = ((void *)0);
  cleanupPorts = 0;
  @throw [Object new];
 }
 @catch(Frob *obj) {
  if(!(0)) abort();
 }
 @catch(id exc) {
  if(!(!receivePort)) abort();
  if(!(!sendPort)) abort();
  if(!(!cleanupPorts)) abort();
 }
}
