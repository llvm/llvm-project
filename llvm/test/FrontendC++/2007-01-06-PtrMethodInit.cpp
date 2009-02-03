// RUN: %llvmgxx %s -emit-llvm -S -o -
// PR1084

extern "C"
{
  typedef unsigned char PRUint8;
  typedef unsigned int PRUint32;
}
typedef PRUint32 nsresult;
struct nsID
{
};
typedef nsID nsIID;
class nsISupports
{
};
extern "C++"
{
  template < class T > struct nsCOMTypeInfo
  {
    static const nsIID & GetIID ()
    {
    }
  };
}

class nsIDOMEvent:public nsISupports
{
};
class nsIDOMEventListener:public nsISupports
{
public:static const nsIID & GetIID ()
  {
  }
  virtual nsresult
    __attribute__ ((regparm (0), cdecl)) HandleEvent (nsIDOMEvent * event) =
    0;
};
class nsIDOMMouseListener:public nsIDOMEventListener
{
public:static const nsIID & GetIID ()
  {
    static const nsIID iid = {
    };
  }
  virtual nsresult
    __attribute__ ((regparm (0),
		    cdecl)) MouseDown (nsIDOMEvent * aMouseEvent) = 0;
};
typedef
typeof (&nsIDOMEventListener::HandleEvent)
  GenericHandler;
     struct EventDispatchData
     {
       PRUint32 message;
       GenericHandler method;
       PRUint8 bits;
     };
     struct EventTypeData
     {
       const EventDispatchData *events;
       int numEvents;
       const nsIID *iid;
     };
     static const EventDispatchData sMouseEvents[] = {
       {
	(300 + 2),
	reinterpret_cast < GenericHandler > (&nsIDOMMouseListener::MouseDown),
	0x01}
     };
static const EventTypeData sEventTypes[] = {
  {
   sMouseEvents, (sizeof (sMouseEvents) / sizeof (sMouseEvents[0])),
   &nsCOMTypeInfo < nsIDOMMouseListener >::GetIID ()}
};
