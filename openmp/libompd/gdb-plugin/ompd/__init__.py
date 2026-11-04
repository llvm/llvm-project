import sys
import os.path
import traceback

if __name__ == "__main__":
    try:
        sys.path.insert(0, os.path.dirname(__file__))

        import ompd

        ompd.main()
        print("OMPD GDB support loaded")
        print("Run 'ompd init' to start debugging")
    except Exception as e:
        traceback.print_exc()
        print("Error: OMPD support could not be loaded", e)
